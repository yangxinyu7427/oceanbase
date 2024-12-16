#include "lib/oblog/ob_log_module.h"
#include "io/easy_io_struct.h"
#include "common/sql_mode/ob_sql_mode.h"
#include "common/ob_range.h"
#include "lib/net/ob_addr.h"
#include "share/ob_define.h"
#include "share/ob_ddl_common.h"
#include "lib/atomic/ob_atomic.h"
#include "lib/ob_name_def.h"
#include "lib/oblog/ob_warning_buffer.h"
#include "lib/list/ob_list.h"
#include "lib/allocator/page_arena.h"
#include "lib/objectpool/ob_pool.h"
#include "lib/time/ob_cur_time.h"
#include "lib/lock/ob_recursive_mutex.h"
#include "lib/hash/ob_link_hashmap.h"
#include "lib/mysqlclient/ob_server_connection_pool.h"
#include "lib/stat/ob_session_stat.h"
#include "rpc/obmysql/ob_mysql_packet.h"
#include "sql/ob_sql_config_provider.h"
#include "sql/ob_end_trans_callback.h"
#include "sql/session/ob_session_val_map.h"
#include "sql/session/ob_basic_session_info.h"
#include "sql/monitor/ob_exec_stat.h"
#include "sql/monitor/ob_security_audit.h"
#include "sql/monitor/ob_security_audit_utils.h"
#include "share/rc/ob_tenant_base.h"
#include "share/rc/ob_context.h"
#include "sql/dblink/ob_dblink_utils.h"
#include "share/resource_manager/ob_cgroup_ctrl.h"
#include "sql/monitor/flt/ob_flt_extra_info.h"
#include "sql/ob_optimizer_trace_impl.h"
#include "sql/monitor/flt/ob_flt_span_mgr.h"
#include "storage/tx/ob_tx_free_route.h"
#include <string>
using namespace oceanbase::common;
using namespace oceanbase::sql;
using namespace oceanbase::common;
using namespace oceanbase::share::schema;
using namespace oceanbase::share;
using namespace oceanbase::pl;
using namespace oceanbase::obmysql;
namespace oceanbase{
namespace sql{
    class PyUDFCache{
    public:
        typedef common::hash::ObHashMap<char*, int, 
            common::hash::NoPthreadDefendMode> PyUdfCacheMapForInt;

        typedef common::hash::ObHashMap<char*, std::shared_ptr<std::string>, 
            common::hash::NoPthreadDefendMode> PyUdfCacheMapForString;

        typedef common::hash::ObHashMap<char*, double, 
            common::hash::NoPthreadDefendMode> PyUdfCacheMapForDouble;

        typedef common::hash::ObHashMap<char*, float*, 
            common::hash::NoPthreadDefendMode> PyUdfCacheMapForMidResult;

        typedef common::hash::ObHashMap<common::ObString, PyUdfCacheMapForString*, 
            common::hash::NoPthreadDefendMode> CacheMapForString;

        typedef common::hash::ObHashMap<common::ObString, PyUdfCacheMapForInt*, 
            common::hash::NoPthreadDefendMode> CacheMapForInt;

        typedef common::hash::ObHashMap<common::ObString, PyUdfCacheMapForDouble*, 
            common::hash::NoPthreadDefendMode> CacheMapForDouble;

        typedef common::hash::ObHashMap<common::ObString, PyUdfCacheMapForMidResult*, 
            common::hash::NoPthreadDefendMode> CacheMapForMidResult;
        
        CacheMapForInt cache_map_for_int_;
        CacheMapForDouble cache_map_for_double_;
        CacheMapForString cache_map_for_string_;
        CacheMapForMidResult cache_map_for_mid_result_;
        std::unordered_set<std::string> udf_list;
        std::unordered_set<std::string> path_list;
        std::unordered_map<std::string,int> mid_res_col_count_map;
    
        // 创建缓存
        int create(){
            int ret = OB_SUCCESS;
            if(OB_FAIL(cache_map_for_int_.create(hash::cal_next_prime(32), ObModIds::OB_HASH_BUCKET, 
                ObModIds::OB_HASH_NODE))){
                //LOG_WARN("create cache_map_for_int failed", K(ret));
            }else if(cache_map_for_string_.create(hash::cal_next_prime(32), ObModIds::OB_HASH_BUCKET, 
                ObModIds::OB_HASH_NODE)){
                //LOG_WARN("create cache_map_for_string failed", K(ret));
            }else if(cache_map_for_double_.create(hash::cal_next_prime(32), ObModIds::OB_HASH_BUCKET, 
                ObModIds::OB_HASH_NODE)){
                //LOG_WARN("create cache_map_for_string failed", K(ret));
            }else if(cache_map_for_mid_result_.create(hash::cal_next_prime(32), ObModIds::OB_HASH_BUCKET, 
                ObModIds::OB_HASH_NODE)){
                //LOG_WARN("create cache_map_for_string failed", K(ret));
            }
            return ret;
        }

        bool find_fine_cache_for_model_path(common::ObString& model_path){
            std::string path_str(model_path.ptr(), model_path.length());
            if(path_list.find(path_str)==path_list.end()){
                return false;
            }
            PyUdfCacheMapForMidResult *cache_map;
            if(cache_map_for_mid_result_.get_refactored(model_path, cache_map)!=OB_SUCCESS){
                return false;
            }
            if(cache_map->empty()){
                return false;
            }
            return true;
        }

        bool find_cache_for_cell(const common::ObString& udf_name){
            std::string udf_str(udf_name.ptr(),udf_name.length());
            if(udf_list.find(udf_str)==udf_list.end()){
                return false;
            }
            return true;
        }

        int create_cache_for_model_path(const common::ObString& model_path){
            int ret = OB_SUCCESS;
            std::string model_str(model_path.ptr(),model_path.length());
            PyUdfCacheMapForMidResult *cache_map=new PyUdfCacheMapForMidResult();
            if(OB_FAIL(cache_map->create(hash::cal_next_prime(500000),
                                              ObModIds::OB_HASH_BUCKET,
                                              ObModIds::OB_HASH_NODE))){
                //LOG_WARN("new_cache_map create fail", K(ret));
            }else if(OB_FAIL(cache_map_for_mid_result_.set_refactored(model_path, cache_map))){
                //LOG_WARN("cache_map_for_int set fail", K(ret));
            }
            path_list.insert(model_str);
            return ret;
        }

        int create_cache_for_cell_for_int(const common::ObString& udf_name){
            int ret = OB_SUCCESS;
            std::string udf_str(udf_name.ptr(),udf_name.length());
            PyUdfCacheMapForInt *cache_map=new PyUdfCacheMapForInt();
            if(OB_FAIL(cache_map->create(hash::cal_next_prime(500000),
                                              ObModIds::OB_HASH_BUCKET,
                                              ObModIds::OB_HASH_NODE))){
                //LOG_WARN("new_cache_map create fail", K(ret));
            }else if(OB_FAIL(cache_map_for_int_.set_refactored(udf_name, cache_map))){
                //LOG_WARN("cache_map_for_int set fail", K(ret));
            }
            udf_list.insert(udf_str);
            return ret;
        }

        int create_cache_for_cell_for_double(const common::ObString& udf_name){
            int ret = OB_SUCCESS;
            std::string udf_str(udf_name.ptr(),udf_name.length());
            PyUdfCacheMapForDouble *cache_map=new PyUdfCacheMapForDouble();
            if(OB_FAIL(cache_map->create(hash::cal_next_prime(500000),
                                              ObModIds::OB_HASH_BUCKET,
                                              ObModIds::OB_HASH_NODE))){
                //LOG_WARN("new_cache_map create fail", K(ret));
            }else if(OB_FAIL(cache_map_for_double_.set_refactored(udf_name, cache_map))){
                //LOG_WARN("cache_map_for_double set fail", K(ret));
            }
            udf_list.insert(udf_str);
            return ret;
        }

        int create_cache_for_cell_for_str(const common::ObString& udf_name){
            int ret = OB_SUCCESS;
            std::string udf_str(udf_name.ptr(),udf_name.length());
            PyUdfCacheMapForString *cache_map=new PyUdfCacheMapForString();
            if(OB_FAIL(cache_map->create(hash::cal_next_prime(500000),
                                              ObModIds::OB_HASH_BUCKET,
                                              ObModIds::OB_HASH_NODE))){
                //LOG_WARN("new_cache_map create fail", K(ret));
            }else if(OB_FAIL(cache_map_for_string_.set_refactored(udf_name, cache_map))){
                //LOG_WARN("cache_map_for_int set fail", K(ret));
            }
            udf_list.insert(udf_str);
            return ret;
        }

        int set_int(const common::ObString& udf_name, string& key, int value){
            int ret = OB_SUCCESS;
            PyUdfCacheMapForInt *cache_map;
            // 有缓存就直接拿出来
            if(OB_FAIL(cache_map_for_int_.get_refactored(udf_name, cache_map))){
                //LOG_WARN("cache_map_for_int get fail", K(ret));
            }
            // 将数据存入map
            size_t length = key.size();
            char* newKey = new char[length + 1];
            std::memcpy(newKey, key.c_str(), length + 1);
            if(OB_FAIL(cache_map->set_refactored(newKey, value))){
                //LOG_WARN("cache_map set fail", K(ret));
            }
            return ret;
        }

        int set_mid_result(const common::ObString& model_path, string& key, float* value){
            int ret = OB_SUCCESS;
            PyUdfCacheMapForMidResult *cache_map;
            // 有缓存就直接拿出来
            if(OB_FAIL(cache_map_for_mid_result_.get_refactored(model_path, cache_map))){
                //LOG_WARN("cache_map_for_int get fail", K(ret));
            }
            // 将数据存入map
            size_t length = key.size();
            char* newKey = new char[length + 1];
            std::memcpy(newKey, key.c_str(), length + 1);
            if(OB_FAIL(cache_map->set_refactored(newKey, value))){
                //LOG_WARN("cache_map set fail", K(ret));
            }
            return ret;
        }

        int set_double(const common::ObString& udf_name, string& key, double value){
            int ret = OB_SUCCESS;
            PyUdfCacheMapForDouble *cache_map;
            // 有缓存就直接拿出来
            if(OB_FAIL(cache_map_for_double_.get_refactored(udf_name, cache_map))){
                //LOG_WARN("cache_map_for_double get fail", K(ret));
            }
            // 将数据存入map
            size_t length = key.size();
            char* newKey = new char[length + 1];
            std::memcpy(newKey, key.c_str(), length + 1);
            if(OB_FAIL(cache_map->set_refactored(newKey, value))){
                //LOG_WARN("cache_map set fail", K(ret));
            }
            return ret;
        }

        int set_string(const common::ObString& udf_name, string& key, string& value){
            int ret = OB_SUCCESS;
            PyUdfCacheMapForString *cache_map;
            // 有缓存就直接拿出来
            if(OB_FAIL(cache_map_for_string_.get_refactored(udf_name, cache_map))){
                //LOG_WARN("cache_map_for_string get fail", K(ret));
            }
            // 将数据存入map
            size_t length = key.size();
            char* newKey = new char[length + 1];
            std::memcpy(newKey, key.c_str(), length + 1);
            if(OB_FAIL(cache_map->set_refactored(newKey, std::make_shared<std::string>(value)))){
                //LOG_WARN("cache_map set fail", K(ret));
            }
            return ret;
        }

        int get_mid_result(const common::ObString& model_path, char* key, float*& value){
            int ret = OB_SUCCESS;
            PyUdfCacheMapForMidResult *cache_map;
            // 有缓存就直接拿出来
            if(OB_FAIL(cache_map_for_mid_result_.get_refactored(model_path, cache_map))){
                //LOG_WARN("cache_map_for_int get fail", K(ret));
            }
            // 将数据从map中取出
            if(OB_FAIL(cache_map->get_refactored(key, value))){
                //LOG_WARN("cache_map set fail", K(ret));
            }
            return ret;
        }

        int get_int(const common::ObString& udf_name, char* key, int& value){
            int ret = OB_SUCCESS;
            PyUdfCacheMapForInt *cache_map;
            // 有缓存就直接拿出来
            if(OB_FAIL(cache_map_for_int_.get_refactored(udf_name, cache_map))){
                //LOG_WARN("cache_map_for_int get fail", K(ret));
            }
            // 将数据从map中取出
            if(OB_FAIL(cache_map->get_refactored(key, value))){
                //LOG_WARN("cache_map set fail", K(ret));
            }
            return ret;
        }

        int get_double(const common::ObString& udf_name, char* key, double& value){
            int ret = OB_SUCCESS;
            PyUdfCacheMapForDouble *cache_map;
            // 有缓存就直接拿出来
            if(OB_FAIL(cache_map_for_double_.get_refactored(udf_name, cache_map))){
                //LOG_WARN("cache_map_for_double get fail", K(ret));
            }
            // 将数据从map中取出
            if(OB_FAIL(cache_map->get_refactored(key, value))){
                //LOG_WARN("cache_map set fail", K(ret));
            }
            return ret;
        }

        int get_string(const common::ObString& udf_name, char* key, string& value){
            int ret = OB_SUCCESS;
            PyUdfCacheMapForString *cache_map;
            // 有缓存就直接拿出来
            if(OB_FAIL(cache_map_for_string_.get_refactored(udf_name, cache_map))){
                //LOG_WARN("cache_map_for_string get fail", K(ret));
            }
            // 将数据存入map
            std::shared_ptr<std::string> value_ptr;
            if(OB_FAIL(cache_map->get_refactored(key, value_ptr))){
                //LOG_WARN("cache_map set fail", K(ret));
            }
            value=*value_ptr.get();
            return ret;
        }

    };

}
}
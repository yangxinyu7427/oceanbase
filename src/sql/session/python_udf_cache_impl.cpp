#define USING_LOG_PREFIX SQL_SESSION
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
        typedef common::hash::ObHashMap<char*, int, 
            common::hash::NoPthreadDefendMode> PyUdfCacheMapForInt;

        typedef common::hash::ObHashMap<char*, std::shared_ptr<std::string>, 
            common::hash::NoPthreadDefendMode> PyUdfCacheMapForString;

        typedef common::hash::ObHashMap<common::ObString, PyUdfCacheMapForString*, 
            common::hash::NoPthreadDefendMode> CacheMapForString;

        typedef common::hash::ObHashMap<common::ObString, PyUdfCacheMapForInt*, 
            common::hash::NoPthreadDefendMode> CacheMapForInt;
        CacheMapForInt cache_map_for_int_;
        CacheMapForString cache_map_for_string_;
        std::unordered_set<std::string> udf_list;
        // 创建缓存
        int create(){
            int ret = OB_SUCCESS;
            if(OB_FAIL(cache_map_for_int_.create(hash::cal_next_prime(32), ObModIds::OB_HASH_BUCKET, 
                ObModIds::OB_HASH_NODE))){
                LOG_WARN("create cache_map_for_int failed", K(ret));
            }else if(cache_map_for_string_.create(hash::cal_next_prime(32), ObModIds::OB_HASH_BUCKET, 
                ObModIds::OB_HASH_NODE)){
                LOG_WARN("create cache_map_for_string failed", K(ret));
            }
            return ret;
        }

        int set_int(const common::ObString& udf_name, char* key, int value){
            int ret = OB_SUCCESS;
            PyUdfCacheMapForInt *cache_map;
            // 判断udf是否已建立相关缓存，如果没建立，就去初始化一个缓存
            std::string udf_str(udf_name.ptr(),udf_name.length());
            if(udf_list.find(udf_str)==udf_list.end()){
                cache_map=new PyUdfCacheMapForInt();
                if(OB_FAIL(cache_map->create(hash::cal_next_prime(500000),
                                              ObModIds::OB_HASH_BUCKET,
                                              ObModIds::OB_HASH_NODE))){
                    LOG_WARN("new_cache_map create fail", K(ret));
                }else if(OB_FAIL(cache_map_for_int_.set_refactored(udf_name, cache_map))){
                    LOG_WARN("cache_map_for_int set fail", K(ret));
                }
                udf_list.insert(udf_str);
            }else{
                // 有缓存就直接拿出来
                if(OB_FAIL(cache_map_for_int_.get_refactored(udf_name, cache_map))){
                    LOG_WARN("cache_map_for_int get fail", K(ret));
                }
            }
            // 将数据存入map
            if(OB_FAIL(cache_map->set_refactored(key, value))){
                LOG_WARN("cache_map set fail", K(ret));
            }
            return ret;
        }

        int set_string(const common::ObString& udf_name, char* key, string& value){
            int ret = OB_SUCCESS;
            PyUdfCacheMapForString *cache_map;
            // 判断udf是否已建立相关缓存，如果没建立，就去初始化一个缓存
            std::string udf_str(udf_name.ptr(),udf_name.length());
            if(udf_list.find(udf_str)==udf_list.end()){
                cache_map=new PyUdfCacheMapForString();
                if(OB_FAIL(cache_map->create(hash::cal_next_prime(500000),
                                              ObModIds::OB_HASH_BUCKET,
                                              ObModIds::OB_HASH_NODE))){
                    LOG_WARN("new_cache_map create fail", K(ret));
                }else if(OB_FAIL(cache_map_for_string_.set_refactored(udf_name, cache_map))){
                    LOG_WARN("cache_map_for_string set fail", K(ret));
                }
                udf_list.insert(udf_str);
            }else{
                // 有缓存就直接拿出来
                if(OB_FAIL(cache_map_for_string_.get_refactored(udf_name, cache_map))){
                    LOG_WARN("cache_map_for_string get fail", K(ret));
                }
            }
            // 将数据存入map
            if(OB_FAIL(cache_map->set_refactored(key, std::make_shared<std::string>(value)))){
                LOG_WARN("cache_map set fail", K(ret));
            }
            return ret;
        }

        int get_int(const common::ObString& udf_name, char* key, int& value){
            int ret = OB_SUCCESS;
            PyUdfCacheMapForInt *cache_map;
            // 判断udf是否已建立相关缓存，如果没建立，就去初始化一个缓存
            std::string udf_str(udf_name.ptr(),udf_name.length());
            if(udf_list.find(udf_str)==udf_list.end()){
                cache_map=new PyUdfCacheMapForInt();
                if(OB_FAIL(cache_map->create(hash::cal_next_prime(500000),
                                              ObModIds::OB_HASH_BUCKET,
                                              ObModIds::OB_HASH_NODE))){
                    LOG_WARN("new_cache_map create fail", K(ret));
                }else if(OB_FAIL(cache_map_for_int_.set_refactored(udf_name, cache_map))){
                    LOG_WARN("cache_map_for_int set fail", K(ret));
                }
                udf_list.insert(udf_str);
            }else{
                // 有缓存就直接拿出来
                if(OB_FAIL(cache_map_for_int_.get_refactored(udf_name, cache_map))){
                    LOG_WARN("cache_map_for_int get fail", K(ret));
                }
            }
            // 将数据从map中取出
            if(OB_FAIL(cache_map->get_refactored(key, value))){
                LOG_WARN("cache_map set fail", K(ret));
            }
            return ret;
        }

        int get_string(const common::ObString& udf_name, char* key, string& value){
            int ret = OB_SUCCESS;
            PyUdfCacheMapForString *cache_map;
            // 判断udf是否已建立相关缓存，如果没建立，就去初始化一个缓存
            std::string udf_str(udf_name.ptr(),udf_name.length());
            if(udf_list.find(udf_str)==udf_list.end()){
                cache_map=new PyUdfCacheMapForString();
                if(OB_FAIL(cache_map->create(hash::cal_next_prime(500000),
                                              ObModIds::OB_HASH_BUCKET,
                                              ObModIds::OB_HASH_NODE))){
                    LOG_WARN("new_cache_map create fail", K(ret));
                }else if(OB_FAIL(cache_map_for_string_.set_refactored(udf_name, cache_map))){
                    LOG_WARN("cache_map_for_string set fail", K(ret));
                }
                udf_list.insert(udf_str);
            }else{
                // 有缓存就直接拿出来
                if(OB_FAIL(cache_map_for_string_.get_refactored(udf_name, cache_map))){
                    LOG_WARN("cache_map_for_string get fail", K(ret));
                }
            }
            // 将数据存入map
            std::shared_ptr<std::string> value_ptr;
            if(OB_FAIL(cache_map->get_refactored(key, value_ptr))){
                LOG_WARN("cache_map set fail", K(ret));
            }
            value=*value_ptr.get();
            return ret;
        }

    };

}
}
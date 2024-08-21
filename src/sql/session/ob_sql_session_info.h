/**
 * Copyright (c) 2021 OceanBase
 * OceanBase CE is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 */

#ifndef OCEANBASE_SQL_SESSION_OB_SQL_SESSION_INFO_
#define OCEANBASE_SQL_SESSION_OB_SQL_SESSION_INFO_

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
#include "sql/monitor/ob_sql_plan_manager.h"
#include "storage/tx/ob_tx_free_route.h"

namespace oceanbase
{
namespace observer
{
class ObQueryDriver;
class ObSqlEndTransCb;
}
namespace pl
{
class ObPLPackageState;
class ObPL;
struct ObPLExecRecursionCtx;
struct ObPLSqlCodeInfo;
class ObPLContext;
class ObDbmsCursorInfo;
} // namespace pl

namespace obmysql
{
class ObMySQLRequestManager;
} // namespace obmysql
namespace share
{
struct ObSequenceValue;
}
using common::ObPsStmtId;
namespace sql
{
class ObResultSet;
class ObPlanCache;
class ObPsCache;
class ObPsSessionInfo;
class ObPsStmtInfo;
class ObStmt;
class ObSQLSessionInfo;
class ObPlanItemMgr;

class SessionInfoKey
{
public:
  SessionInfoKey() : sessid_(0), proxy_sessid_(0) { }
  SessionInfoKey(uint32_t sessid, uint64_t proxy_sessid = 0) : sessid_(sessid), proxy_sessid_(proxy_sessid) {}
  uint64_t hash() const
  { uint64_t hash_value = 0;
    hash_value = common::murmurhash(&sessid_, sizeof(sessid_), hash_value);
    return hash_value;
  };
  int compare(const SessionInfoKey & r)
  {
    int cmp = 0;
    if (sessid_ < r.sessid_) {
      cmp = -1;
    } else if (sessid_ > r.sessid_) {
      cmp = 1;
    } else {
      cmp = 0;
    }
    return cmp;
  }
public:
  uint32_t sessid_;
  uint64_t proxy_sessid_; //不参与compare, 仅存储值用于ObCTASCleanUp遍历
};

struct ObContextUnit
{
  inline void free(common::ObIAllocator &alloc) {
    alloc.free(value_.ptr());
    alloc.free(attribute_.ptr());
  }
  int deep_copy(const common::ObString &attribute,
                const common::ObString &value,
                common::ObIAllocator &alloc) {
    int ret = OB_SUCCESS;
    if (OB_FAIL(ob_write_string(alloc, attribute, attribute_))) {
      SQL_ENG_LOG(WARN, "failed to copy attribute", K(ret));
    } else if (OB_FAIL(ob_write_string(alloc, value, value_))) {
      alloc.free(attribute_.ptr());
      SQL_ENG_LOG(WARN, "failed to copy value", K(ret));
    }
    return ret;
  }
  ObString attribute_;
  ObString value_;
  TO_STRING_KV(K(attribute_), K(value_));
  OB_UNIS_VERSION(1);
};

struct ObSessionEncryptInfo
{
  // 用于sql传给事务层接口中缓存当前租户是否加密.
  // 缓存在session的目的是为了性能.防止频繁访问租户配置项.
  bool is_encrypt_;
  uint64_t last_modify_time_;
  ObSessionEncryptInfo() : is_encrypt_(false), last_modify_time_(0) {}
  inline void reset()
  {
    is_encrypt_ = false;
    last_modify_time_ = 0;
  }
};

struct ObSessionStat final
{
  ObSessionStat() : total_logical_read_(0), total_physical_read_(0), total_logical_write_(0),
                    total_lock_count_(0), total_cpu_time_us_(0), total_exec_time_us_(0),
                    total_alive_time_us_(0)
      {}
  void reset() { new (this) ObSessionStat(); }

  TO_STRING_KV(K_(total_logical_read), K_(total_physical_read), K_(total_logical_write),
      K_(total_lock_count), K_(total_cpu_time_us), K_(total_exec_time_us), K_(total_alive_time_us));

  uint64_t total_logical_read_;
  uint64_t total_physical_read_;
  uint64_t total_logical_write_;
  uint64_t total_lock_count_;
  uint64_t total_cpu_time_us_;
  uint64_t total_exec_time_us_;
  uint64_t total_alive_time_us_;
};

//该结构的并发控制跟Session上的其他变量一样
class ObTenantCachedSchemaGuardInfo
{
public:
  ObTenantCachedSchemaGuardInfo()
    : schema_guard_(share::schema::ObSchemaMgrItem::MOD_CACHED_GUARD)
  { reset(); }
  ~ObTenantCachedSchemaGuardInfo() { reset(); }
  void reset();

  // 缓存的信息，可能跟schema service维护的version不一致，
  // 调用者需要自己跟情况决定是否调用refresh_tenant_schema_version接口
  share::schema::ObSchemaGetterGuard &get_schema_guard() { return schema_guard_; }
  int refresh_tenant_schema_guard(const uint64_t tenant_id);

  // 尝试将schema_mgr的ref进行归还，规则：每隔10s对schema_guard做一次revert操作；
  // 1. 如果session 有请求访问，则每条语句结束之后，尝试 触发一次；
  // 2. 如果session 没有频发的访问，通过session_mgr后台的遍历来解决；
  void try_revert_schema_guard();
private:
  share::schema::ObSchemaGetterGuard schema_guard_;
  // 记录重新获取schema guard的时间 戳，后台 需要具备兜底revert guard ref的机制，避免schema mgr槽位无法释放
  int64_t ref_ts_;
  uint64_t tenant_id_;
  int64_t schema_version_;
};

enum SessionSyncInfoType {
  //SESSION_SYNC_USER_VAR,  // for user variables
  SESSION_SYNC_APPLICATION_INFO = 0, // for application info
  SESSION_SYNC_APPLICATION_CONTEXT = 1, // for app ctx
  SESSION_SYNC_CLIENT_ID = 2, // for client identifier
  SESSION_SYNC_CONTROL_INFO = 3, // for full trace link control info
  SESSION_SYNC_SYS_VAR = 4,   // for system variables
  SESSION_SYNC_TXN_STATIC_INFO = 5,       // 5: basic txn info
  SESSION_SYNC_TXN_DYNAMIC_INFO = 6,      // 6: txn dynamic info
  SESSION_SYNC_TXN_PARTICIPANTS_INFO = 7, // 7: txn dynamic info
  SESSION_SYNC_TXN_EXTRA_INFO = 8,        // 8: txn dynamic info
  SESSION_SYNC_MAX_TYPE,
};


struct SessSyncTxnTypeSet {
  SessSyncTxnTypeSet() {
    types_.add_member(SessionSyncInfoType::SESSION_SYNC_TXN_STATIC_INFO);
    types_.add_member(SessionSyncInfoType::SESSION_SYNC_TXN_DYNAMIC_INFO);
    types_.add_member(SessionSyncInfoType::SESSION_SYNC_TXN_PARTICIPANTS_INFO);
    types_.add_member(SessionSyncInfoType::SESSION_SYNC_TXN_EXTRA_INFO);
  };
  common::ObFixedBitSet<oceanbase::sql::SessionSyncInfoType::SESSION_SYNC_MAX_TYPE> types_;
  bool is_contain(const int t) { return types_.has_member(t); }
  void type_range(int &min, int &max) {
    min = SessionSyncInfoType::SESSION_SYNC_TXN_STATIC_INFO;
    max = SessionSyncInfoType::SESSION_SYNC_TXN_EXTRA_INFO;
  }
  static SessSyncTxnTypeSet &get_instance() {
    static SessSyncTxnTypeSet instance;
    return instance;
  }
};

class ObSessInfoEncoder {
public:
  ObSessInfoEncoder() : is_changed_(false) {}
  virtual ~ObSessInfoEncoder() {}
  virtual int serialize(ObSQLSessionInfo &sess, char *buf, const int64_t length, int64_t &pos) = 0;
  virtual int deserialize(ObSQLSessionInfo &sess, const char *buf, const int64_t length, int64_t &pos) = 0;
  virtual int64_t get_serialize_size(ObSQLSessionInfo& sess) const = 0;
  bool is_changed_;
};

class ObSysVarEncoder : public ObSessInfoEncoder {
public:
  ObSysVarEncoder():ObSessInfoEncoder() {}
  ~ObSysVarEncoder() {}
  int serialize(ObSQLSessionInfo &sess, char *buf, const int64_t length, int64_t &pos);
  int deserialize(ObSQLSessionInfo &sess, const char *buf, const int64_t length, int64_t &pos);
  int64_t get_serialize_size(ObSQLSessionInfo& sess) const;
};

class ObAppInfoEncoder : public ObSessInfoEncoder {
public:
  ObAppInfoEncoder():ObSessInfoEncoder() {}
  ~ObAppInfoEncoder() {}
  int serialize(ObSQLSessionInfo &sess, char *buf, const int64_t length, int64_t &pos);
  int deserialize(ObSQLSessionInfo &sess, const char *buf, const int64_t length, int64_t &pos);
  int64_t get_serialize_size(ObSQLSessionInfo& sess) const;
  int set_client_info(ObSQLSessionInfo* sess, const ObString &client_info);
  int set_module_name(ObSQLSessionInfo* sess, const ObString &mod);
  int set_action_name(ObSQLSessionInfo* sess, const ObString &act);
};

//class ObUserVarEncoder : public ObSessInfoEncoder {
//public:
//  ObUserVarEncoder():ObSessInfoEncoder() {}
//  ~ObUserVarEncoder() {}
//  int serialize(ObBasicSessionInfo &sess, char *buf, const int64_t length, int64_t &pos);
//  int deserialize(ObBasicSessionInfo &sess, const char *buf, const int64_t length, int64_t &pos);
//  int64_t get_serialize_size(ObBasicSessionInfo& sess) const;
//  // implements of other variables need to monitor
//};

class ObAppCtxInfoEncoder : public ObSessInfoEncoder {
public:
  ObAppCtxInfoEncoder() : ObSessInfoEncoder() {}
  virtual ~ObAppCtxInfoEncoder() {}
  virtual int serialize(ObSQLSessionInfo &sess, char *buf, const int64_t length, int64_t &pos) override;
  virtual int deserialize(ObSQLSessionInfo &sess, const char *buf, const int64_t length, int64_t &pos) override;
  virtual int64_t get_serialize_size(ObSQLSessionInfo& sess) const override;
};
class ObClientIdInfoEncoder : public ObSessInfoEncoder {
public:
  ObClientIdInfoEncoder() : ObSessInfoEncoder() {}
  virtual ~ObClientIdInfoEncoder() {}
  virtual int serialize(ObSQLSessionInfo &sess, char *buf, const int64_t length, int64_t &pos) override;
  virtual int deserialize(ObSQLSessionInfo &sess, const char *buf, const int64_t length, int64_t &pos) override;
  virtual int64_t get_serialize_size(ObSQLSessionInfo &sess) const override;
};

class ObControlInfoEncoder : public ObSessInfoEncoder {
public:
  ObControlInfoEncoder() : ObSessInfoEncoder() {}
  virtual ~ObControlInfoEncoder() {}
  virtual int serialize(ObSQLSessionInfo &sess, char *buf, const int64_t length, int64_t &pos) override;
  virtual int deserialize(ObSQLSessionInfo &sess, const char *buf, const int64_t length, int64_t &pos) override;
  virtual int64_t get_serialize_size(ObSQLSessionInfo &sess) const override;
  static const int16_t CONINFO_BY_SESS = 0xC078;
};

#define DEF_SESSION_TXN_ENCODER(CLS)                                    \
class CLS final : public ObSessInfoEncoder {                            \
public:                                                                 \
  int serialize(ObSQLSessionInfo &sess, char *buf, const int64_t length, int64_t &pos) override; \
  int deserialize(ObSQLSessionInfo &sess, const char *buf, const int64_t length, int64_t &pos) override; \
  int64_t get_serialize_size(ObSQLSessionInfo &sess) const override;    \
};
DEF_SESSION_TXN_ENCODER(ObTxnStaticInfoEncoder);
DEF_SESSION_TXN_ENCODER(ObTxnDynamicInfoEncoder);
DEF_SESSION_TXN_ENCODER(ObTxnParticipantsInfoEncoder);
DEF_SESSION_TXN_ENCODER(ObTxnExtraInfoEncoder);

#undef DEF_SESSION_TXN_ENCODER

typedef common::hash::ObHashMap<uint64_t, pl::ObPLPackageState *,
                                common::hash::NoPthreadDefendMode> ObPackageStateMap;

typedef common::hash::ObHashMap<char*, int,
                                common::hash::NoPthreadDefendMode> ObSinglePyUdfFunCacheMap; 

typedef common::hash::ObHashMap<char*, char*,
                                common::hash::NoPthreadDefendMode> ObSinglePyUdfStrFunCacheMap; 

typedef common::hash::ObHashMap<common::ObString, ObSinglePyUdfFunCacheMap*,
                                common::hash::NoPthreadDefendMode> ObPyUdfFunCacheMap;

typedef common::hash::ObHashMap<common::ObString, bool,
                                common::hash::NoPthreadDefendMode> ObHistoryPyUdfMap;

typedef common::hash::ObHashMap<common::ObString, ObSinglePyUdfStrFunCacheMap*,
                                common::hash::NoPthreadDefendMode> ObPyUdfStrFunCacheMap;

typedef common::hash::ObHashMap<uint64_t, share::ObSequenceValue,
                                common::hash::NoPthreadDefendMode> ObSequenceCurrvalMap;
typedef common::hash::ObHashMap<common::ObString,
                                ObContextUnit *,
                                common::hash::NoPthreadDefendMode,
                                common::hash::hash_func<common::ObString>,
                                common::hash::equal_to<common::ObString>,
                                common::hash::SimpleAllocer<typename common::hash::HashMapTypes<common::ObString, ObContextUnit *>::AllocType>,
                                common::hash::NormalPointer,
                                oceanbase::common::ObMalloc,
                                2> ObInnerContextHashMap;
struct ObInnerContextMap {
  ObInnerContextMap(common::ObIAllocator &alloc) : context_name_(),
                    context_map_(nullptr), alloc_(alloc) {}
  void destroy()
  {
    if (OB_NOT_NULL(context_map_)) {
      for (auto it = context_map_->begin(); it != context_map_->end(); ++it) {
        it->second->free(alloc_);
        alloc_.free(it->second);
      }
    }
    destroy_map();
  }
  void destroy_map()
  {
    if (OB_NOT_NULL(context_map_)) {
      context_map_->destroy();
      alloc_.free(context_map_);
    }
    if (OB_NOT_NULL(context_name_.ptr())) {
      alloc_.free(context_name_.ptr());
    }
  }
  int init()
  {
    int ret = OB_SUCCESS;
    if (OB_ISNULL(context_map_ = static_cast<ObInnerContextHashMap *>
                                  (alloc_.alloc(sizeof(ObInnerContextHashMap))))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      SQL_ENG_LOG(WARN, "failed to alloc mem for hash map", K(ret));
    } else {
      new (context_map_) ObInnerContextHashMap ();
      if (OB_FAIL(context_map_->create(hash::cal_next_prime(32),
                                      ObModIds::OB_HASH_BUCKET,
                                      ObModIds::OB_HASH_NODE))) {
        SQL_ENG_LOG(WARN, "failed to init hash map", K(ret));
      }
    }
    return ret;
  }
  ObString context_name_;
  ObInnerContextHashMap *context_map_;
  common::ObIAllocator &alloc_;
  OB_UNIS_VERSION(1);
};
typedef common::hash::ObHashMap<common::ObString, ObInnerContextMap *,
                                common::hash::NoPthreadDefendMode,
                                common::hash::hash_func<common::ObString>,
                                common::hash::equal_to<common::ObString>,
                                common::hash::SimpleAllocer<typename common::hash::HashMapTypes<common::ObString, ObInnerContextMap *>::AllocType>,
                                common::hash::NormalPointer,
                                oceanbase::common::ObMalloc,
                                2> ObContextsMap;
typedef common::LinkHashNode<SessionInfoKey> SessionInfoHashNode;
typedef common::LinkHashValue<SessionInfoKey> SessionInfoHashValue;
// ObBasicSessionInfo存储系统变量及其相关变量，并存储远程执行SQL task时，需要序列化到远端的状态
// ObPsInfoMgr存储prepared statement相关信息
// ObSQLSessionInfo存储其他执行时状态信息，远程执行SQL执行计划时，**不需要**序列化到远端
class ObSQLSessionInfo: public common::ObVersionProvider, public ObBasicSessionInfo, public SessionInfoHashValue
{
  OB_UNIS_VERSION(1);
public:
  friend class LinkExecCtxGuard;
  // notice!!! register exec ctx to session for later access
  // used for temp session, such as session for rpc processor, px worker, das processor, etc
  // not used for main session
  class ExecCtxSessionRegister
  {
  public:
    ExecCtxSessionRegister(ObSQLSessionInfo &session, ObExecContext &exec_ctx)
    {
      session.set_cur_exec_ctx(&exec_ctx);
    }
  };
  friend class ExecCtxSessionRegister;
  enum SessionType
  {
    INVALID_TYPE,
    USER_SESSION,
    INNER_SESSION
  };
  // for switch stmt.
  class StmtSavedValue : public ObBasicSessionInfo::StmtSavedValue
  {
  public:
    StmtSavedValue()
      : ObBasicSessionInfo::StmtSavedValue()
    {
      reset();
    }
    inline void reset()
    {
      ObBasicSessionInfo::StmtSavedValue::reset();
      audit_record_.reset();
      session_type_ = INVALID_TYPE;
      inner_flag_ = false;
      is_ignore_stmt_ = false;
    }
  public:
    ObAuditRecordData audit_record_;
    SessionType session_type_;
    bool inner_flag_;
    bool is_ignore_stmt_;
  };

  class CursorCache {
    public:
      CursorCache() : mem_context_(nullptr), next_cursor_id_(1LL << 31), pl_cursor_map_() {}
      virtual ~CursorCache() { NULL != mem_context_ ? DESTROY_CONTEXT(mem_context_) : (void)(NULL); }
      int init(uint64_t tenant_id)
      {
        int ret = OB_SUCCESS;
        if (OB_FAIL(ROOT_CONTEXT->CREATE_CONTEXT(mem_context_,
            lib::ContextParam().set_mem_attr(tenant_id, ObModIds::OB_PL)))) {
          SQL_ENG_LOG(WARN, "create memory entity failed");
        } else if (OB_ISNULL(mem_context_)) {
          ret = OB_ERR_UNEXPECTED;
          SQL_ENG_LOG(WARN, "null memory entity returned");
        } else if (!pl_cursor_map_.created() &&
                   OB_FAIL(pl_cursor_map_.create(common::hash::cal_next_prime(32),
                                                 ObModIds::OB_HASH_BUCKET, ObModIds::OB_HASH_NODE))) {
          SQL_ENG_LOG(WARN, "create sequence current value map failed", K(ret));
        } else { /*do nothing*/ }
        return ret;
      }
      int close_all(sql::ObSQLSessionInfo &session)
      {
        int ret = OB_SUCCESS;
        common::ObSEArray<uint64_t, 32> cursor_ids;
        ObSessionStatEstGuard guard(session.get_effective_tenant_id(), session.get_sessid());
        for (CursorMap::iterator iter = pl_cursor_map_.begin();  //ignore ret
            iter != pl_cursor_map_.end();
            ++iter) {
          pl::ObPLCursorInfo *cursor_info = iter->second;
          if (OB_ISNULL(cursor_info)) {
            ret = OB_ERR_UNEXPECTED;
            SQL_ENG_LOG(WARN, "cursor info is NULL", K(cursor_info), K(ret));
          } else {
            cursor_ids.push_back(cursor_info->get_id());
          }
        }
        for (int64_t i = 0; i < cursor_ids.count(); i++) {
          uint64_t cursor_id = cursor_ids.at(i);
          if (OB_FAIL(session.close_cursor(cursor_id))) {
            SQL_ENG_LOG(WARN, "failed to close cursor",
                        K(cursor_id), K(session.get_sessid()), K(ret));
          } else {
            SQL_ENG_LOG(INFO, "NOTICE: cursor is closed unexpectedly",
                        K(cursor_id), K(session.get_sessid()), K(ret));
          }
        }
        return ret;
      }
      inline bool is_inited() const { return NULL != mem_context_; }
      void reset()
      {
        pl_cursor_map_.reuse();
        next_cursor_id_ = 1LL << 31;
        if (NULL != mem_context_) {
          DESTROY_CONTEXT(mem_context_);
          mem_context_ = NULL;
        }
      }
      inline int64_t gen_cursor_id() { return __sync_add_and_fetch(&next_cursor_id_, 1); }
    public:
      lib::MemoryContext mem_context_;
      int64_t next_cursor_id_;
      typedef common::hash::ObHashMap<int64_t, pl::ObPLCursorInfo*,
                                      common::hash::NoPthreadDefendMode> CursorMap;
      CursorMap pl_cursor_map_;
  };

  class ObCachedTenantConfigInfo
  {
  public:
    ObCachedTenantConfigInfo(ObSQLSessionInfo *session) :
                                 is_external_consistent_(false),
                                 enable_batched_multi_statement_(false),
                                 enable_sql_extension_(false),
                                 saved_tenant_info_(0),
                                 enable_bloom_filter_(true),
                                 px_join_skew_handling_(true),
                                 px_join_skew_minfreq_(30),
                                 at_type_(ObAuditTrailType::NONE),
                                 sort_area_size_(128*1024*1024),
                                 print_sample_ppm_(0),
                                 last_check_ec_ts_(0),
                                 session_(session)
    {
    }
    ~ObCachedTenantConfigInfo() {}
    void refresh();
    bool get_is_external_consistent() const { return is_external_consistent_; }
    bool get_enable_batched_multi_statement() const { return enable_batched_multi_statement_; }
    bool get_enable_bloom_filter() const { return enable_bloom_filter_; }
    bool get_enable_sql_extension() const { return enable_sql_extension_; }
    ObAuditTrailType get_at_type() const { return at_type_; }
    int64_t get_sort_area_size() const { return ATOMIC_LOAD(&sort_area_size_); }
    int64_t get_print_sample_ppm() const { return ATOMIC_LOAD(&print_sample_ppm_); }
    bool get_px_join_skew_handling() const { return px_join_skew_handling_; }
    int64_t get_px_join_skew_minfreq() const { return px_join_skew_minfreq_; }
  private:
    //租户级别配置项缓存session 上，避免每次获取都需要刷新
    bool is_external_consistent_;
    bool enable_batched_multi_statement_;
    bool enable_sql_extension_;
    uint64_t saved_tenant_info_;
    bool enable_bloom_filter_;
    bool px_join_skew_handling_;
    int64_t px_join_skew_minfreq_;
    ObAuditTrailType at_type_;
    int64_t sort_area_size_;
    // for record sys config print_sample_ppm
    int64_t print_sample_ppm_;
    int64_t last_check_ec_ts_;
    ObSQLSessionInfo *session_;
  };

  class ApplicationInfo {
    OB_UNIS_VERSION(1);
  public:
    common::ObString module_name_;  // name as set by the dbms_application_info(set_module)
    common::ObString action_name_;  // action as set by the dbms_application_info(set_action)
    common::ObString client_info_;  // for addition info
    void reset() {
      module_name_.reset();
      action_name_.reset();
      client_info_.reset();
    }
    TO_STRING_KV(K_(module_name), K_(action_name), K_(client_info));
  };

public:
  ObSQLSessionInfo();
  virtual ~ObSQLSessionInfo();

  int init(uint32_t sessid, uint64_t proxy_sessid,
           common::ObIAllocator *bucket_allocator,
           const ObTZInfoMap *tz_info = NULL,
           int64_t sess_create_time = 0,
           uint64_t tenant_id = OB_INVALID_TENANT_ID);
  void destroy_session_plan_mgr();
  //for test
  int test_init(uint32_t version, uint32_t sessid, uint64_t proxy_sessid,
           common::ObIAllocator *bucket_allocator);
  void destroy(bool skip_sys_var = false);
  void reset(bool skip_sys_var);
  void clean_status();
  void set_plan_cache(ObPlanCache *cache) { plan_cache_ = cache; }
  void set_ps_cache(ObPsCache *cache) { ps_cache_ = cache; }
  const common::ObWarningBuffer &get_show_warnings_buffer() const { return show_warnings_buf_; }
  const common::ObWarningBuffer &get_warnings_buffer() const { return warnings_buf_; }
  common::ObWarningBuffer &get_warnings_buffer() { return warnings_buf_; }
  void reset_warnings_buf()
  {
    warnings_buf_.reset();
    pl_exact_err_msg_.reset();
  }
  void restore_auto_commit()
  {
    if (restore_auto_commit_) {
      set_autocommit(true);
      restore_auto_commit_ = false;
    }
  }
  void set_restore_auto_commit() { restore_auto_commit_ = true; }
  void reset_show_warnings_buf() { show_warnings_buf_.reset(); }
  ObPrivSet get_user_priv_set() const { return user_priv_set_; }
  ObPrivSet get_db_priv_set() const { return db_priv_set_; }
  ObPlanCache *get_plan_cache();
  ObPlanCache *get_plan_cache_directly() const { return plan_cache_; };
  ObPsCache *get_ps_cache();
  obmysql::ObMySQLRequestManager *get_request_manager();
  sql::ObFLTSpanMgr *get_flt_span_manager();
  ObPlanItemMgr *get_sql_plan_manager();
  ObPlanItemMgr *get_plan_table_manager();
  void set_user_priv_set(const ObPrivSet priv_set) { user_priv_set_ = priv_set; }
  void set_db_priv_set(const ObPrivSet priv_set) { db_priv_set_ = priv_set; }
  void set_show_warnings_buf(int error_code);
  void update_show_warnings_buf();
  void set_global_sessid(const int64_t global_sessid)
  {
    global_sessid_ = global_sessid;
  }
  oceanbase::sql::ObDblinkCtxInSession &get_dblink_context() { return dblink_context_; }
  void set_sql_request_level(int64_t sql_req_level) { sql_req_level_ = sql_req_level; }
  int64_t get_sql_request_level() { return sql_req_level_; }
  int64_t get_next_sql_request_level() { return sql_req_level_ + 1; }
  int64_t get_global_sessid() const { return global_sessid_; }
  void set_read_uncommited(bool read_uncommited) { read_uncommited_ = read_uncommited; }
  bool get_read_uncommited() const { return read_uncommited_; }
  void set_version_provider(const common::ObVersionProvider *version_provider)
  {
    version_provider_ = version_provider;
  }
  const common::ObVersion get_frozen_version() const
  {
    return version_provider_->get_frozen_version();
  }
  const common::ObVersion get_merged_version() const
  {
    return version_provider_->get_merged_version();
  }
  void set_config_provider(const ObSQLConfigProvider *config_provider)
  {
    config_provider_ = config_provider;
  }
  bool is_read_only() const { return config_provider_->is_read_only(); };
  int64_t get_nlj_cache_limit() const { return config_provider_->get_nlj_cache_limit(); };
  bool is_terminate(int &ret) const;

  void set_curr_trans_start_time(int64_t t) { curr_trans_start_time_ = t; };
  int64_t get_curr_trans_start_time() const { return curr_trans_start_time_; };

  void set_curr_trans_last_stmt_time(int64_t t) { curr_trans_last_stmt_time_ = t; };
  int64_t get_curr_trans_last_stmt_time() const { return curr_trans_last_stmt_time_; };

  void set_sess_create_time(const int64_t t) { sess_create_time_ = t; };
  int64_t get_sess_create_time() const { return sess_create_time_; };

  void set_last_refresh_temp_table_time(const int64_t t) { last_refresh_temp_table_time_ = t; };
  int64_t get_last_refresh_temp_table_time() const { return last_refresh_temp_table_time_; };

  void set_has_temp_table_flag() { has_temp_table_flag_ = true; };
  bool get_has_temp_table_flag() const { return has_temp_table_flag_; };
  void set_accessed_session_level_temp_table() { has_accessed_session_level_temp_table_ = true; }
  bool has_accessed_session_level_temp_table() const { return has_accessed_session_level_temp_table_; }
  // 清除临时表
  int drop_temp_tables(const bool is_sess_disconn = true, const bool is_xa_trans = false);
  void refresh_temp_tables_sess_active_time(); //更新临时表的sess active time
  int drop_reused_oracle_temp_tables();
  int delete_from_oracle_temp_tables(const obrpc::ObDropTableArg &const_drop_table_arg);

  void set_for_trigger_package(bool value) { is_for_trigger_package_ = value; }
  bool is_for_trigger_package() const { return is_for_trigger_package_; }
  void set_trans_type(transaction::ObTxClass t) { trans_type_ = t; }
  transaction::ObTxClass get_trans_type() const { return trans_type_; }

  void get_session_priv_info(share::schema::ObSessionPrivInfo &session_priv) const;
  void set_found_rows(const int64_t count) { found_rows_ = count; }
  int64_t get_found_rows() const { return found_rows_; }
  void set_affected_rows(const int64_t count) { affected_rows_ = count; }
  int64_t get_affected_rows() const { return affected_rows_; }
  bool has_user_super_privilege() const;
  bool has_user_process_privilege() const;
  int check_read_only_privilege(const bool read_only,
                                const ObSqlTraits &sql_traits);
  int check_global_read_only_privilege(const bool read_only,
                                       const ObSqlTraits &sql_traits);

  int remove_prepare(const ObString &ps_name);
  int get_prepare_id(const ObString &ps_name, ObPsStmtId &ps_id) const;
  int add_prepare(const ObString &ps_name, ObPsStmtId ps_id);
  int remove_ps_session_info(const ObPsStmtId stmt_id);
  int get_ps_session_info(const ObPsStmtId stmt_id,
                          ObPsSessionInfo *&ps_session_info) const;
  int64_t get_ps_session_info_size() const { return ps_session_info_map_.size(); }
  inline pl::ObPL *get_pl_engine() const { return GCTX.pl_engine_; }

  pl::ObPLCursorInfo *get_pl_implicit_cursor();
  const pl::ObPLCursorInfo *get_pl_implicit_cursor() const;

  pl::ObPLSqlCodeInfo *get_pl_sqlcode_info();
  const pl::ObPLSqlCodeInfo *get_pl_sqlcode_info() const;

  bool has_pl_implicit_savepoint();
  void clear_pl_implicit_savepoint();
  void set_has_pl_implicit_savepoint(bool v);

  inline pl::ObPLContext *get_pl_context() { return pl_context_; }
  inline const pl::ObPLContext *get_pl_context() const { return pl_context_; }
  inline void set_pl_stack_ctx(pl::ObPLContext *pl_stack_ctx)
  {
    pl_context_ = pl_stack_ctx;
  }

  bool is_pl_debug_on();

  inline void set_pl_attached_id(uint32_t id) { pl_attach_session_id_ = id; }
  inline uint32_t get_pl_attached_id() const { return pl_attach_session_id_; }

  inline common::hash::ObHashSet<common::ObString> *get_pl_sync_pkg_vars()
  {
    return pl_sync_pkg_vars_;
  }

  inline void set_pl_query_sender(observer::ObQueryDriver *driver) { pl_query_sender_ = driver; }
  inline observer::ObQueryDriver* get_pl_query_sender() { return pl_query_sender_; }

  inline void set_ps_protocol(bool is_ps_protocol) { pl_ps_protocol_ = is_ps_protocol; }
  inline bool is_ps_protocol() { return pl_ps_protocol_; }

  inline void set_ob20_protocol(bool is_20protocol) { is_ob20_protocol_ = is_20protocol; }
  inline bool is_ob20_protocol() { return is_ob20_protocol_; }

  inline void set_session_var_sync(bool is_session_var_sync)
              { is_session_var_sync_ = is_session_var_sync; }
  inline bool is_session_var_sync() { return is_session_var_sync_; }

  int replace_user_variable(const common::ObString &name, const ObSessionVariable &value);
  int replace_user_variable(
    ObExecContext &ctx, const common::ObString &name, const ObSessionVariable &value);
  int replace_user_variables(const ObSessionValMap &user_var_map);
  int replace_user_variables(ObExecContext &ctx, const ObSessionValMap &user_var_map);
  int set_package_variables(ObExecContext &ctx, const ObSessionValMap &user_var_map);
  int set_package_variable(ObExecContext &ctx,
    const common::ObString &key, const common::ObObj &value, bool from_proxy = false);

  inline bool get_pl_can_retry() { return pl_can_retry_; }
  inline void set_pl_can_retry(bool can_retry) { pl_can_retry_ = can_retry; }

  CursorCache &get_cursor_cache() { return pl_cursor_cache_; }
  pl::ObPLCursorInfo *get_cursor(int64_t cursor_id);
  pl::ObDbmsCursorInfo *get_dbms_cursor(int64_t cursor_id);
  int add_cursor(pl::ObPLCursorInfo *cursor);
  int close_cursor(pl::ObPLCursorInfo *&cursor);
  int close_cursor(int64_t cursor_id);
  int make_cursor(pl::ObPLCursorInfo *&cursor);
  int init_cursor_cache();
  int make_dbms_cursor(pl::ObDbmsCursorInfo *&cursor,
                       uint64_t id = OB_INVALID_ID);
  int close_dbms_cursor(int64_t cursor_id);
  int print_all_cursor();

  inline void *get_inner_conn() { return inner_conn_; }
  inline void set_inner_conn(void *inner_conn)
  {
    inner_conn_ = inner_conn;
  }

  // show trace
  common::ObTraceEventRecorder *get_trace_buf();
  void clear_trace_buf();

  ObEndTransAsyncCallback &get_end_trans_cb() { return end_trans_cb_; }
  observer::ObSqlEndTransCb &get_mysql_end_trans_cb()
  {
    return end_trans_cb_.get_mysql_end_trans_cb();
  }
  int get_collation_type_of_names(const ObNameTypeClass type_class, common::ObCollationType &cs_type) const;
  int name_case_cmp(const common::ObString &name, const common::ObString &name_other,
                    const ObNameTypeClass type_class, bool &is_equal) const;
  int kill_query();
  int set_query_deadlocked();

  inline void set_inner_session()
  {
    inner_flag_ = true;
    session_type_ = INNER_SESSION;
  }
  inline void set_user_session()
  {
    inner_flag_ = false;
    session_type_ = USER_SESSION;
  }
  void set_session_type_with_flag();
  void set_session_type(SessionType session_type) { session_type_ = session_type; }
  inline SessionType get_session_type() const { return session_type_; }
  // sql from obclient, proxy, PL are all marked as user_session
  // NOTE: for sql from PL, is_inner() = true, is_user_session() = true
  inline bool is_user_session() const { return USER_SESSION == session_type_; }
  void set_early_lock_release(bool enable);
  bool get_early_lock_release() const { return enable_early_lock_release_; }

  bool is_inner() const
  {
    return inner_flag_;
  }
  void reset_audit_record(bool need_retry = false)
  {
    if (!need_retry) {
      audit_record_.reset();
    } else {
      // memset without try_cnt_ and exec_timestamp_
      int64_t try_cnt = audit_record_.try_cnt_;
      ObExecTimestamp exec_timestamp = audit_record_.exec_timestamp_;
      audit_record_.reset();
      audit_record_.try_cnt_ = try_cnt;
      audit_record_.exec_timestamp_ = exec_timestamp;
    }
  }
  ObAuditRecordData &get_raw_audit_record() { return audit_record_; }
  //在最最终需要push record到audit buffer中时使用该方法，
  //该方法会将一些session中能够拿到的并且重试过程中不会变化的
  //字段初始化
  const ObAuditRecordData &get_final_audit_record(ObExecuteMode mode);
  ObSessionStat &get_session_stat() { return session_stat_; }
  void update_stat_from_audit_record();
  void update_alive_time_stat();
  void handle_audit_record(bool need_retry, ObExecuteMode exec_mode);

  void set_is_remote(bool is_remote) { is_remote_session_ = is_remote; }
  bool is_remote_session() const { return is_remote_session_; }

  int save_session(StmtSavedValue &saved_value);
  int save_sql_session(StmtSavedValue &saved_value);
  int restore_sql_session(StmtSavedValue &saved_value);
  int restore_session(StmtSavedValue &saved_value);
  ObExecContext *get_cur_exec_ctx() { return cur_exec_ctx_; }

  int begin_nested_session(StmtSavedValue &saved_value, bool skip_cur_stmt_tables = false);
  int end_nested_session(StmtSavedValue &saved_value);

  //package state related
  inline  ObPackageStateMap &get_package_state_map() { return package_state_map_; }
  inline int get_package_state(uint64_t package_id, pl::ObPLPackageState *&package_state)
  {
    return package_state_map_.get_refactored(package_id, package_state);
  }
  inline int add_package_state(uint64_t package_id, pl::ObPLPackageState *package_state)
  {
    return package_state_map_.set_refactored(package_id, package_state);
  }
  inline int del_package_state(uint64_t package_id)
  {
    return package_state_map_.erase_refactored(package_id);
  }
  void reset_pl_debugger_resource();
  void reset_all_package_changed_info();
  void reset_all_package_state();
  int reset_all_serially_package_state();
  bool is_package_state_changed() const;
  bool get_changed_package_state_num() const;
  int add_changed_package_info(ObExecContext &exec_ctx);
  int shrink_package_info();

  // 当前 session 上发生的 sequence.nextval 读取 sequence 值，
  // 都会由 ObSequence 算子将读取结果保存在当前 session 上
  int get_sequence_value(uint64_t tenant_id,
                         uint64_t seq_id,
                         share::ObSequenceValue &value);
  int set_sequence_value(uint64_t tenant_id,
                         uint64_t seq_id,
                         const share::ObSequenceValue &value);

  int get_context_values(const common::ObString &context_name,
                        const common::ObString &attribute,
                        common::ObString &value,
                        bool &exist);
  int set_context_values(const common::ObString &context_name,
                        const common::ObString &attribute,
                        const common::ObString &value);
  int clear_all_context(const common::ObString &context_name);
  int clear_context(const common::ObString &context_name,
                    const common::ObString &attribute);
  int64_t get_curr_session_context_size() const { return curr_session_context_size_; }
  void reuse_context_map() 
  {
    for (auto it = contexts_map_.begin(); it != contexts_map_.end(); ++it) {
      if (OB_NOT_NULL(it->second)) {
        it->second->destroy();
      }
    }
    contexts_map_.reuse();
    curr_session_context_size_ = 0;
  }

  int set_client_id(const common::ObString &client_identifier);

  bool has_sess_info_modified() const;
  int set_module_name(const common::ObString &mod);
  int set_action_name(const common::ObString &act);
  int set_client_info(const common::ObString &client_info);
  ApplicationInfo& get_client_app_info() { return client_app_info_; }
  int get_sess_encoder(const SessionSyncInfoType sess_sync_info_type, ObSessInfoEncoder* &encoder);
  const common::ObString& get_module_name() const { return client_app_info_.module_name_; }
  const common::ObString& get_action_name() const  { return client_app_info_.action_name_; }
  const common::ObString& get_client_info() const { return client_app_info_.client_info_; }
  const FLTControlInfo& get_control_info() const { return flt_control_info_; }
  FLTControlInfo& get_control_info() { return flt_control_info_; }
  void set_flt_control_info(const FLTControlInfo &con_info);
  bool is_send_control_info() { return is_send_control_info_; }
  void set_send_control_info(bool is_send) { is_send_control_info_ = is_send; }
  bool is_coninfo_set_by_sess() { return coninfo_set_by_sess_; }
  void set_coninfo_set_by_sess(bool is_set_by_sess) { coninfo_set_by_sess_ = is_set_by_sess; }
  bool is_trace_enable() { return trace_enable_; }
  void set_trace_enable(bool trace_enable) { trace_enable_ = trace_enable; }
  bool is_auto_flush_trace() {return auto_flush_trace_;}
  void set_auto_flush_trace(bool auto_flush_trace) { auto_flush_trace_ = auto_flush_trace; }
  ObSysVarEncoder& get_sys_var_encoder() { return sys_var_encoder_; }
  //ObUserVarEncoder& get_usr_var_encoder() { return usr_var_encoder_; }
  ObAppInfoEncoder& get_app_info_encoder() { return app_info_encoder_; }
  ObAppCtxInfoEncoder &get_app_ctx_encoder() { return app_ctx_info_encoder_; }
  ObClientIdInfoEncoder &get_client_info_encoder() { return client_id_info_encoder_;}
  ObControlInfoEncoder &get_control_info_encoder() { return control_info_encoder_;}
  ObContextsMap &get_contexts_map() { return contexts_map_; }
  ObPyUdfFunCacheMap &get_pyudf_funcache_map() { return pyudf_funcache_map_; }
  ObHistoryPyUdfMap &get_history_pyudf_map() { return history_pyudf_map_; }
  ObPyUdfStrFunCacheMap &get_pyudf_str_funcache_map() { return pyudf_str_funcache_map_; }
  int get_mem_ctx_alloc(common::ObIAllocator *&alloc);
  int update_sess_sync_info(const SessionSyncInfoType sess_sync_info_type,
                                const char *buf, const int64_t length, int64_t &pos);
  int prepare_ps_stmt(const ObPsStmtId inner_stmt_id,
                      const ObPsStmtInfo *stmt_info,
                      ObPsStmtId &client_stmt_id,
                      bool &already_exists,
                      bool is_inner_sql);
  int get_inner_ps_stmt_id(ObPsStmtId cli_stmt_id, ObPsStmtId &inner_stmt_id);
  int close_ps_stmt(ObPsStmtId stmt_id);

  bool is_encrypt_tenant();
  int64_t get_expect_group_id() const { return expect_group_id_; }
  void set_expect_group_id(int64_t group_id) { expect_group_id_ = group_id; }
	bool get_group_id_not_expected() const { return group_id_not_expected_; }
  void set_group_id_not_expected(bool value) { group_id_not_expected_ = value; }
  int is_force_temp_table_inline(bool &force_inline) const;
  int is_force_temp_table_materialize(bool &force_materialize) const;
  int is_temp_table_transformation_enabled(bool &transformation_enabled) const;
  int is_groupby_placement_transformation_enabled(bool &transformation_enabled) const;
  bool is_in_range_optimization_enabled() const;

  ObSessionDDLInfo &get_ddl_info() { return ddl_info_; }
  void set_ddl_info(const ObSessionDDLInfo &ddl_info) { ddl_info_ = ddl_info; }
  bool is_table_name_hidden() const { return is_table_name_hidden_; }
  void set_table_name_hidden(const bool is_hidden) { is_table_name_hidden_ = is_hidden; }

  ObTenantCachedSchemaGuardInfo &get_cached_schema_guard_info() { return cached_schema_guard_info_; }
  int set_enable_role_array(const common::ObIArray<uint64_t> &role_id_array);
  common::ObIArray<uint64_t>& get_enable_role_array() { return enable_role_array_; }
  void set_in_definer_named_proc(bool in_proc) {in_definer_named_proc_ = in_proc; }
  bool get_in_definer_named_proc() {return in_definer_named_proc_; }
  bool get_prelock() { return prelock_; }
  void set_prelock(bool prelock) { prelock_ = prelock; }

  void set_priv_user_id(uint64_t priv_user_id) { priv_user_id_ = priv_user_id; }
  uint64_t get_priv_user_id() {
    return (priv_user_id_ == OB_INVALID_ID) ? get_user_id() : priv_user_id_; }
  int64_t get_xa_end_timeout_seconds() const;
  int set_xa_end_timeout_seconds(int64_t seconds);
  uint64_t get_priv_user_id_allow_invalid() { return priv_user_id_; }
  int get_xa_last_result() const { return xa_last_result_; }
  void set_xa_last_result(const int result) { xa_last_result_ = result; }
  // 为了性能优化考虑，租户级别配置项不需要实时获取，缓存在session上，每隔5s触发一次刷新
  void refresh_tenant_config() { cached_tenant_config_info_.refresh(); }
  bool is_support_external_consistent()
  {
    cached_tenant_config_info_.refresh();
    return cached_tenant_config_info_.get_is_external_consistent();
  }
  bool is_enable_batched_multi_statement()
  {
    cached_tenant_config_info_.refresh();
    return cached_tenant_config_info_.get_enable_batched_multi_statement();
  }
  bool is_enable_bloom_filter()
  {
    cached_tenant_config_info_.refresh();
    return cached_tenant_config_info_.get_enable_bloom_filter();
  }
  int64_t get_px_join_skew_minfreq()
  {
    cached_tenant_config_info_.refresh();
    return cached_tenant_config_info_.get_px_join_skew_minfreq();
  }
  bool get_px_join_skew_handling()
  {
    cached_tenant_config_info_.refresh();
    return cached_tenant_config_info_.get_px_join_skew_handling();
  }

  bool is_enable_sql_extension()
  {
    cached_tenant_config_info_.refresh();
    return cached_tenant_config_info_.get_enable_sql_extension();
  }
  bool is_ps_prepare_stage() const { return is_ps_prepare_stage_; }
  void set_is_ps_prepare_stage(bool v) { is_ps_prepare_stage_ = v; }
  int get_tenant_audit_trail_type(ObAuditTrailType &at_type)
  {
    cached_tenant_config_info_.refresh();
    at_type = cached_tenant_config_info_.get_at_type();
    return common::OB_SUCCESS;
  }
  int64_t get_tenant_sort_area_size()
  {
    cached_tenant_config_info_.refresh();
    return cached_tenant_config_info_.get_sort_area_size();
  }
  int64_t get_tenant_print_sample_ppm()
  {
    cached_tenant_config_info_.refresh();
    return cached_tenant_config_info_.get_print_sample_ppm();
  }
  int get_tmp_table_size(uint64_t &size);
  int ps_use_stream_result_set(bool &use_stream);
  void set_proxy_version(uint64_t v) { proxy_version_ = v; }
  uint64_t get_proxy_version() { return proxy_version_; }

  void set_ignore_stmt(bool v) { is_ignore_stmt_ = v; }
  bool is_ignore_stmt() const { return is_ignore_stmt_; }

  // piece
  void *get_piece_cache(bool need_init = false);

  void set_load_data_exec_session(bool v) { is_load_data_exec_session_ = v; }
  bool is_load_data_exec_session() const { return is_load_data_exec_session_; }
  inline ObSqlString &get_pl_exact_err_msg() { return pl_exact_err_msg_; }
  void set_got_tenant_conn_res(bool v) { got_tenant_conn_res_ = v; }
  bool has_got_tenant_conn_res() const { return got_tenant_conn_res_; }
  void set_got_user_conn_res(bool v) { got_user_conn_res_ = v; }
  bool has_got_user_conn_res() const { return got_user_conn_res_; }
  void set_conn_res_user_id(uint64_t v) { conn_res_user_id_ = v; }
  uint64_t get_conn_res_user_id() const { return conn_res_user_id_; }
  int on_user_connect(share::schema::ObSessionPrivInfo &priv_info, const ObUserInfo *user_info);
  int on_user_disconnect();
  virtual void reset_tx_variable();
  ObOptimizerTraceImpl& get_optimizer_tracer() { return optimizer_tracer_; }
public:
  bool has_tx_level_temp_table() const { return tx_desc_ && tx_desc_->with_temporary_table(); }
private:
  int close_all_ps_stmt();
  void destroy_contexts_map(ObContextsMap &map, common::ObIAllocator &alloc);
  inline int init_mem_context(uint64_t tenant_id);
  void set_cur_exec_ctx(ObExecContext *cur_exec_ctx) { cur_exec_ctx_ = cur_exec_ctx; }

  static const int64_t MAX_STORED_PLANS_COUNT = 10240;
  static const int64_t MAX_IPADDR_LENGTH = 64;
private:
  bool is_inited_;
  // store the warning message from the most recent statement in the current session
  common::ObWarningBuffer warnings_buf_;
  common::ObWarningBuffer show_warnings_buf_;
  sql::ObEndTransAsyncCallback end_trans_cb_;
  ObAuditRecordData audit_record_;

  ObPrivSet user_priv_set_;
  ObPrivSet db_priv_set_;
  int64_t curr_trans_start_time_;
  int64_t curr_trans_last_stmt_time_;
  int64_t sess_create_time_;  //会话创建时间, 目前仅用于临时表的清理判断
  int64_t last_refresh_temp_table_time_; //会话最后一次刷新临时表sess active time时间, 仅用于proxy连接模式
  bool has_temp_table_flag_;  //会话是否创建过临时表
  bool has_accessed_session_level_temp_table_;  //是否访问过Session临时表
  bool enable_early_lock_release_;
  // trigger.
  bool is_for_trigger_package_;
  transaction::ObTxClass trans_type_;
  const common::ObVersionProvider *version_provider_;
  const ObSQLConfigProvider *config_provider_;
  char tenant_buff_[sizeof(share::ObTenantSpaceFetcher)];
  obmysql::ObMySQLRequestManager *request_manager_;
  sql::ObFLTSpanMgr *flt_span_mgr_;
  ObPlanItemMgr *sql_plan_manager_;
  ObPlanItemMgr *plan_table_manager_;
  ObPlanCache *plan_cache_;
  ObPsCache *ps_cache_;
  //记录select stmt中scan出来的结果集行数，供设置sql_calc_found_row时，found_row()使用；
  int64_t found_rows_;
  //记录dml操作中affected_row，供row_count()使用
  int64_t affected_rows_;
  int64_t global_sessid_;
  bool read_uncommited_; //记录当前语句是否读取了未提交的修改
  common::ObTraceEventRecorder *trace_recorder_;
  //标记一个事务中是否有write的操作，在设置read_only时，用来判断commit语句是否可以成功
  // if has_write_stmt_in_trans_ && read_only => can't not commit
  // else can commit
  // in_transaction_ has been merged into trans_flags_.
//  int64_t has_write_stmt_in_trans_;
  bool inner_flag_; // 是否为内部请求的虚拟session
  // 2.2版本之后，不再使用该变量
  bool is_max_availability_mode_;
  typedef common::hash::ObHashMap<ObPsStmtId, ObPsSessionInfo *,
                                  common::hash::NoPthreadDefendMode> PsSessionInfoMap;
  PsSessionInfoMap ps_session_info_map_;
  inline int try_create_ps_session_info_map()
  {
    int ret = OB_SUCCESS;
    static const int64_t PS_BUCKET_NUM = 64;
    if (OB_UNLIKELY(!ps_session_info_map_.created())) {
      ret = ps_session_info_map_.create(common::hash::cal_next_prime(PS_BUCKET_NUM),
                                        common::ObModIds::OB_HASH_BUCKET_PS_SESSION_INFO,
                                        common::ObModIds::OB_HASH_NODE_PS_SESSION_INFO);
    }
    return ret;
  }

  typedef common::hash::ObHashMap<common::ObString, ObPsStmtId,
                                  common::hash::NoPthreadDefendMode> PsNameIdMap;
  PsNameIdMap ps_name_id_map_;
  inline int try_create_ps_name_id_map()
  {
    int ret = OB_SUCCESS;
    static const int64_t PS_BUCKET_NUM = 64;
    if (OB_UNLIKELY(!ps_name_id_map_.created())) {
      ret = ps_name_id_map_.create(common::hash::cal_next_prime(PS_BUCKET_NUM),
                                   common::ObModIds::OB_HASH_BUCKET_PS_SESSION_INFO,
                                   common::ObModIds::OB_HASH_NODE_PS_SESSION_INFO);
    }
    return ret;
  }

  ObPsStmtId next_client_ps_stmt_id_;
  bool is_remote_session_;//用来记录是否为执行远程计划创建的session
  SessionType session_type_;
  ObPackageStateMap package_state_map_;
  ObSequenceCurrvalMap sequence_currval_map_;
  ObContextsMap contexts_map_;
public:
  ObPyUdfFunCacheMap pyudf_funcache_map_;
  ObHistoryPyUdfMap history_pyudf_map_;
  ObPyUdfStrFunCacheMap pyudf_str_funcache_map_;
private:
  int64_t curr_session_context_size_;

  pl::ObPLContext *pl_context_;
  CursorCache pl_cursor_cache_;
  // if any commit executed, the PL block can not be retried as a whole.
  // otherwise the PL block can be retried in all.
  // if false == pl_can_retry_, we can only retry query in PL blocks locally
  bool pl_can_retry_; //标记当前执行的PL是否可以整体重试

  uint32_t pl_attach_session_id_; // 如果当前session执行过dbms_debug.attach_session, 记录目标session的ID

  observer::ObQueryDriver *pl_query_sender_; // send query result in mysql pl
  bool pl_ps_protocol_; // send query result use this protocol
  bool is_ob20_protocol_; // mark as whether use oceanbase 2.0 protocol

  bool is_session_var_sync_; //session var sync support flag.

  int64_t last_plan_id_; // 记录上一个计划的 plan_id，用于 show trace 中显示 sql 物理计划

  common::hash::ObHashSet<common::ObString> *pl_sync_pkg_vars_ = NULL;

  void *inner_conn_;  // ObInnerSQLConnection * will cause .h included from each other.

  ObSessionStat session_stat_;

  ObSessionEncryptInfo encrypt_info_;

  common::ObSEArray<uint64_t, 8> enable_role_array_;
  ObTenantCachedSchemaGuardInfo cached_schema_guard_info_;
  bool in_definer_named_proc_;
  uint64_t priv_user_id_;
  int64_t xa_end_timeout_seconds_;
  int xa_last_result_;
  // 为了性能优化考虑，租户级别配置项不需要实时获取，缓存在session上，每隔5s触发一次刷新
  ObCachedTenantConfigInfo cached_tenant_config_info_;
  bool prelock_;
  uint64_t proxy_version_;
  uint64_t min_proxy_version_ps_; // proxy大于该版本时，相同sql返回不同的Stmt id
  //新引擎表达式类型推导的时候需要通过ignore_stmt来确定cast_mode，
  //由于很多接口拿不到stmt里的ignore flag,只能通过session来传递，这里只能在生成计划阶段使用
  //在CG后这个状态将被清空
  bool is_ignore_stmt_;
  ObSessionDDLInfo ddl_info_;
  bool is_table_name_hidden_;
  void *piece_cache_;
  bool is_load_data_exec_session_;
  ObSqlString pl_exact_err_msg_;
  bool is_ps_prepare_stage_;
  // Record whether this session has got connection resource, which means it increased connections count.
  // It's used for on_user_disconnect.
  // No matter whether apply for resource successfully, a session will call on_user_disconnect when disconnect.
  // While only session got connection resource can release connection resource and decrease connections count.
  bool got_tenant_conn_res_;
  bool got_user_conn_res_;
  uint64_t conn_res_user_id_;
  bool tx_level_temp_table_;
  // get_session_allocator can only apply for fixed-length memory.
  // To customize the memory length, you need to use malloc_alloctor of mem_context
  lib::MemoryContext mem_context_;
  ApplicationInfo client_app_info_;
  char module_buf_[common::OB_MAX_MOD_NAME_LENGTH];
  char action_buf_[common::OB_MAX_ACT_NAME_LENGTH];
  char client_info_buf_[common::OB_MAX_CLIENT_INFO_LENGTH];
  FLTControlInfo flt_control_info_;
  bool trace_enable_ = false;
  bool is_send_control_info_ = false;  // whether send control info to client
  bool auto_flush_trace_ = false;
  bool coninfo_set_by_sess_ = false;

  ObSessInfoEncoder* sess_encoders_[SESSION_SYNC_MAX_TYPE] = {
                            //&usr_var_encoder_,
                            &app_info_encoder_,
                            &app_ctx_info_encoder_,
                            &client_id_info_encoder_,
                            &control_info_encoder_,
                            &sys_var_encoder_,
                            &txn_static_info_encoder_,
                            &txn_dynamic_info_encoder_,
                            &txn_participants_info_encoder_,
                            &txn_extra_info_encoder_
                            };
  ObSysVarEncoder sys_var_encoder_;
  //ObUserVarEncoder usr_var_encoder_;
  ObAppInfoEncoder app_info_encoder_;
  ObAppCtxInfoEncoder app_ctx_info_encoder_;
  ObClientIdInfoEncoder client_id_info_encoder_;
  ObControlInfoEncoder control_info_encoder_;
  ObTxnStaticInfoEncoder txn_static_info_encoder_;
  ObTxnDynamicInfoEncoder txn_dynamic_info_encoder_;
  ObTxnParticipantsInfoEncoder txn_participants_info_encoder_;
  ObTxnExtraInfoEncoder txn_extra_info_encoder_;
public:
  void post_sync_session_info();
  void prep_txn_free_route_baseline(bool reset_audit = true);
  void set_txn_free_route(bool txn_free_route);
  int calc_txn_free_route();
  bool can_txn_free_route() const;
  virtual bool is_txn_free_route_temp() const { return tx_desc_ != NULL && txn_free_route_ctx_.is_temp(*tx_desc_); }
  transaction::ObTxnFreeRouteCtx &get_txn_free_route_ctx() { return txn_free_route_ctx_; }
  uint64_t get_txn_free_route_flag() const { return txn_free_route_ctx_.get_audit_record(); }
  void check_txn_free_route_alive();
private:
  transaction::ObTxnFreeRouteCtx txn_free_route_ctx_;
  //save the current sql exec context in session
  //and remove the record when the SQL execution ends
  //in order to access exec ctx through session during SQL execution
  ObExecContext *cur_exec_ctx_;
  bool restore_auto_commit_; // for dblink xa transaction to restore the value of auto_commit
  oceanbase::sql::ObDblinkCtxInSession dblink_context_;
  int64_t sql_req_level_; // for sql request between cluster avoid dead lock, such as dblink dead lock
  int64_t expect_group_id_;
  // When try packet retry failed, set this flag true and retry at current thread.
  // This situation is unexpected and will report a warning to user.
  bool group_id_not_expected_;
  ObOptimizerTraceImpl optimizer_tracer_;
};

inline bool ObSQLSessionInfo::is_terminate(int &ret) const
{
  bool bret = false;
  if (QUERY_KILLED == get_session_state()) {
    bret = true;
    SQL_ENG_LOG(WARN, "query interrupted session",
                "query", get_current_query_string(),
                "key", get_sessid(),
                "proxy_sessid", get_proxy_sessid());
    ret = common::OB_ERR_QUERY_INTERRUPTED;
  } else if (QUERY_DEADLOCKED == get_session_state()) {
    bret = true;
    SQL_ENG_LOG(WARN, "query deadlocked",
                "query", get_current_query_string(),
                "key", get_sessid(),
                "proxy_sessid", get_proxy_sessid());
    ret = common::OB_DEAD_LOCK;
  } else if (SESSION_KILLED == get_session_state()) {
    bret = true;
    ret = common::OB_ERR_SESSION_INTERRUPTED;
  }
  return bret;
}

} // namespace sql
} // namespace oceanbase

#endif // OCEANBASE_SQL_SESSION_OB_SQL_SESSION_INFO_

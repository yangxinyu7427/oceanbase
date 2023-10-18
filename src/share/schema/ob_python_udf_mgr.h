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

#ifndef OB_PYTHON_UDF_MGR_H_
#define OB_PYTHON_UDF_MGR_H_

#include "lib/hash/ob_hashmap.h"
#include "lib/hash/ob_pointer_hashmap.h"
#include "lib/container/ob_vector.h"
#include "lib/allocator/page_arena.h"
#include "share/schema/ob_python_udf.h"

namespace oceanbase
{
namespace share
{
namespace schema
{

class ObSimpleModelSchema : public ObSchema
{
public:
  ObSimpleModelSchema();
  explicit ObSimpleModelSchema(common::ObIAllocator *allocator);
  ObSimpleModelSchema(const ObSimpleModelSchema &src_schema);
  virtual ~ObSimpleModelSchema();
  ObSimpleModelSchema &operator =(const ObSimpleModelSchema &other);
  TO_STRING_KV(K_(tenant_id),
               K_(model_id),
               K_(model_name),
               K_(arg_num),
               K_(arg_names),
               K_(arg_types),
               K_(ret),
               K_(pycall),
               K_(schema_version));
  virtual void reset();
  inline bool is_valid() const;
  inline int64_t get_convert_size() const;
  inline void set_tenant_id(const uint64_t tenant_id) { tenant_id_ = tenant_id; }
  inline uint64_t get_tenant_id() const { return tenant_id_; }
  inline void set_model_id(const uint64_t model_id) { model_id_ = model_id; }
  inline uint64_t get_model_id() const { return model_id_; }
  inline void set_schema_version(const int64_t schema_version) { schema_version_ = schema_version; }
  inline int64_t get_schema_version() const { return schema_version_; }
  inline int set_model_name(const common::ObString &name)
  { return deep_copy_str(name, model_name_); }
  inline const char *get_model_name() const { return extract_str(model_name_); }
  inline const common::ObString &get_model_name_str() const { return model_name_; }
  inline common::ObString get_model_name_str() { return model_name_; }
  inline ObTenantModelId get_tenant_model_id() const
  { return ObTenantModelId(tenant_id_, model_name_); }

  //the ObSimpleUDFSchema must have the same interface just like ObUDF.
  //ObSchemaRetrieveUtils's template function will use these interface.
  inline int set_name(const common::ObString &name) { return deep_copy_str(name, model_name_); }
  inline void set_ret(const enum ObPythonUDF::PyUdfRetType ret) { ret_ = ret; }
  inline void set_ret(const int ret) { ret_ = ObPythonUDF::PyUdfRetType(ret); }
  inline void set_arg_num(const int arg_num) { arg_num_ = arg_num; }
  void set_arg_names(const char* arg_name);
  int set_arg_names(const common::ObString arg_name);
  void set_arg_types(const std::string type_string);
  int set_arg_types(const common::ObString type_string);
  inline int set_pycall(const common::ObString &pycall) { return deep_copy_str(pycall, pycall_); }

  inline const char *get_name() const { return extract_str(model_name_); }
  inline const common::ObString &get_name_str() const { return model_name_; }
  inline int get_arg_num() const { return arg_num_; }
  const char *get_arg_names() const { return extract_str(arg_names_); }
  inline const common::ObString &get_arg_names_str() const { return arg_names_; }
  inline const char *get_arg_types() const { return extract_str(arg_types_); }
  inline const common::ObString &get_arg_types_str() const { return arg_types_; }
  inline enum ObPythonUDF::PyUdfRetType get_ret() const { return ret_; }
  inline const char *get_pycall() const { return extract_str(pycall_); }
  inline const common::ObString &get_pycall_str() const { return pycall_; }

private:
  uint64_t tenant_id_;
  uint64_t model_id_;
  common::ObString model_name_;
  int arg_num_;
  common::ObString arg_names_;
  common::ObString arg_types_;
  enum ObPythonUDF::PyUdfRetType ret_;
  common::ObString pycall_;
  int64_t schema_version_;
};

template<class T, class V>
struct ObGetPythonUDFKey {
  void operator()(const T &t, const V &v) const {
    UNUSED(t);
    UNUSED(v);
  }
};

class ObPythonUDFHashWrapper
{
public:
  ObPythonUDFHashWrapper()
    : tenant_id_(common::OB_INVALID_ID),
      model_name_() {}
  ObPythonUDFHashWrapper(uint64_t tenant_id, const common::ObString &model_name)
    : tenant_id_(tenant_id),
      model_name_(model_name) {}
  ~ObPythonUDFHashWrapper() {}
  inline uint64_t hash() const
  {
    uint64_t hash_ret = 0;
    hash_ret = common::murmurhash(&tenant_id_, sizeof(uint64_t), 0);
    hash_ret = common::murmurhash(get_model_name().ptr(), get_model_name().length(), hash_ret);
    return hash_ret;
  }
  inline bool operator==(const ObPythonUDFHashWrapper &rv) const{
    return (tenant_id_ == rv.get_tenant_id())
        && (model_name_ == rv.get_model_name());
  }
  inline void set_tenant_id(uint64_t tenant_id) { tenant_id_ = tenant_id; }
  inline void set_model_name(const common::ObString &model_name) { model_name_ = model_name; }
  inline uint64_t get_tenant_id() const { return tenant_id_; }
  inline const common::ObString &get_model_name() const { return model_name_; }
  TO_STRING_KV(K_(tenant_id), K_(model_name));

private:
  uint64_t tenant_id_;
  common::ObString model_name_;
};

template<>
struct ObGetPythonUDFKey<ObPythonUDFHashWrapper, ObSimpleModelSchema *>
{
  ObPythonUDFHashWrapper operator() (const ObSimpleModelSchema * udf) const {
    ObPythonUDFHashWrapper hash_wrap;
    if (!OB_ISNULL(udf)) {
      hash_wrap.set_tenant_id(udf->get_tenant_id());
      hash_wrap.set_model_name(udf->get_model_name());
    }
    return hash_wrap;
  }
};


class ObPythonUDFMgr
{
public:
  typedef common::ObSortedVector<ObSimpleModelSchema *> UDFInfos;
  typedef common::hash::ObPointerHashMap<ObPythonUDFHashWrapper, ObSimpleModelSchema *, ObGetPythonUDFKey> ObUDFMap;
  typedef UDFInfos::iterator UDFIter;
  typedef UDFInfos::const_iterator ConstUDFIter;
  ObPythonUDFMgr();
  explicit ObPythonUDFMgr(common::ObIAllocator &allocator);
  virtual ~ObPythonUDFMgr();
  int init();
  void reset();
  ObPythonUDFMgr &operator =(const ObPythonUDFMgr &other);
  int assign(const ObPythonUDFMgr &other);
  int deep_copy(const ObPythonUDFMgr &other);
  void dump() const;
  int get_model_schema_count(int64_t &udf_schema_count) const;
  int get_schema_statistics(ObSchemaStatisticsInfo &schema_info) const;
  int add_model(const ObSimpleModelSchema &udf_schema);
  int add_models(const common::ObIArray<ObSimpleModelSchema> &udf_schema);
  int del_model(const ObTenantModelId &udf);

  int get_model_schema(const uint64_t model_id,
                     const ObSimpleModelSchema *&udf_schema) const;
  int get_model_info_version(uint64_t model_id, int64_t &model_version) const;
  int get_model_schema_with_name(const uint64_t tenant_id,
                               const common::ObString &name,
                               const ObSimpleModelSchema *&model_schema) const;
  int get_model_schemas_in_tenant(const uint64_t tenant_id,
      common::ObIArray<const ObSimpleModelSchema *> &udf_schemas) const;
  int del_schemas_in_tenant(const uint64_t tenant_id);
  inline static bool compare_model(const ObSimpleModelSchema *lhs,
                                 const ObSimpleModelSchema *rhs) {
    return lhs->get_tenant_model_id() < rhs->get_tenant_model_id();
  }
  inline static bool equal_model(const ObSimpleModelSchema *lhs,
                               const ObSimpleModelSchema *rhs) {
    return lhs->get_tenant_model_id() == rhs->get_tenant_model_id();
  }
  static int rebuild_model_hashmap(const UDFInfos &udf_infos,
                                 ObUDFMap &udf_map);
private:
  inline static bool compare_with_tenant_model_id(const ObSimpleModelSchema *lhs,
                                                const ObTenantModelId &tenant_outline_id);
  inline static bool equal_to_tenant_model_id(const ObSimpleModelSchema *lhs,
                                            const ObTenantModelId &tenant_outline_id);
private:
  bool is_inited_;
  common::ObArenaAllocator local_allocator_;
  common::ObIAllocator &allocator_;
  UDFInfos udf_infos_;
  ObUDFMap udf_map_;
};


} //end of schema
} //end of share
} //end of oceanbase

#endif
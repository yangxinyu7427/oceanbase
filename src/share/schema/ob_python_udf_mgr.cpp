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

#define USING_LOG_PREFIX SHARE_SCHEMA
#include "ob_python_udf_mgr.h"
#include "lib/allocator/ob_mod_define.h"
#include "lib/oblog/ob_log.h"
#include "lib/oblog/ob_log_module.h"
#include "share/schema/ob_schema_utils.h"

namespace oceanbase
{
namespace share
{
namespace schema
{
using namespace std;
using namespace common;
using namespace common::hash;

ObSimpleModelSchema::ObSimpleModelSchema()
  : ObSchema(), tenant_id_(common::OB_INVALID_ID), model_id_(common::OB_INVALID_ID), model_name_(), arg_num_(0),
arg_names_(), arg_types_(), ret_(ObPythonUDF::UDF_UNINITIAL), pycall_(), schema_version_(common::OB_INVALID_VERSION) 
{
  reset();
}

ObSimpleModelSchema::ObSimpleModelSchema(ObIAllocator *allocator)
  : ObSchema(allocator), tenant_id_(common::OB_INVALID_ID), model_id_(common::OB_INVALID_ID), model_name_(), arg_num_(0),
arg_names_(), arg_types_(), ret_(ObPythonUDF::UDF_UNINITIAL), pycall_(), schema_version_(common::OB_INVALID_VERSION) 
{
  reset();
}

ObSimpleModelSchema::ObSimpleModelSchema(const ObSimpleModelSchema &other)
  : ObSchema(), tenant_id_(common::OB_INVALID_ID), model_id_(common::OB_INVALID_ID), model_name_(), arg_num_(0),
arg_names_(), arg_types_(), ret_(ObPythonUDF::UDF_UNINITIAL), pycall_(), schema_version_(common::OB_INVALID_VERSION) 
{
  reset();
  *this = other;
}

ObSimpleModelSchema::~ObSimpleModelSchema()
{
}

void ObSimpleModelSchema::reset()
{
  tenant_id_ = OB_INVALID_ID;
  model_name_.reset();
  arg_names_.reset();
  arg_types_.reset();
  ret_ = ObPythonUDF::UDF_UNINITIAL;
  pycall_.reset();
  ObSchema::reset();
}

bool ObSimpleModelSchema::is_valid() const
{
  bool ret = true;
  if (OB_INVALID_ID == tenant_id_ ||
      schema_version_ < 0 ||
      model_name_.empty()) {
    ret = false;
  }
  return ret;
}

int64_t ObSimpleModelSchema::get_convert_size() const
{
  int64_t convert_size = 0;

  convert_size += sizeof(ObSimpleModelSchema);
  convert_size += model_name_.length() + 1;
  convert_size += arg_names_.length() + 1;
  convert_size += arg_types_.length() + 1;
  convert_size += pycall_.length() + 1;
  
  return convert_size;
}

ObSimpleModelSchema &ObSimpleModelSchema::operator =(const ObSimpleModelSchema &other)
{
  if (this != &other) {
    reset();
    int ret = OB_SUCCESS;
    error_ret_ = other.error_ret_;
    tenant_id_ = other.tenant_id_;
    model_id_ = other.model_id_;
    arg_num_ = other.arg_num_;
    schema_version_ = other.schema_version_;
    ret_ = other.ret_;
    if (OB_FAIL(deep_copy_str(other.model_name_, model_name_))) {
      LOG_WARN("Fail to deep copy model name", K(ret));
    } else if (OB_FAIL(deep_copy_str(other.arg_names_, arg_names_))) {
      LOG_WARN("Fail to deep copy arg names", K(ret));
    } else if (OB_FAIL(deep_copy_str(other.arg_types_, arg_types_))) {
      LOG_WARN("Fail to deep copy arg types", K(ret));
    } else if (OB_FAIL(deep_copy_str(other.pycall_, pycall_))) {
      LOG_WARN("Fail to deep copy pycall", K(ret));
    }
    if (OB_FAIL(ret)) {
      error_ret_ = ret;
    }
  }
  return *this;
}

void ObSimpleModelSchema::set_arg_names(const char* arg_name) {
    char* arg_names_str = const_cast<char*>(get_arg_names());
    arg_names_.reset();
    std::string result = arg_names_str ? arg_names_str : "";
    if (!result.empty()) {
        //不同的参数名之间用逗号分隔
        result += ",";
    }
    //添加新的arg_name
    result += arg_name;
    //将结果转回ObString
    deep_copy_str(common::ObString(result.length(), result.c_str()),arg_names_);
}

void ObSimpleModelSchema::set_arg_types(const std::string type_string) {
    char* arg_types_str = const_cast<char*>(get_arg_types());
    arg_types_.reset();
    std::string result = arg_types_str ? arg_types_str : "";
    if (!result.empty()){
        //不同的参数类型之间用逗号分隔
        result += ",";
    }
    //添加新的arg_type
    result += type_string;
    deep_copy_str(common::ObString(result.length(), result.c_str()),arg_types_);
}

//////////////////////////////////////////////////////////////////////////


ObPythonUDFMgr::ObPythonUDFMgr() :
    is_inited_(false),
    local_allocator_(ObModIds::OB_SCHEMA_GETTER_GUARD),
    allocator_(local_allocator_),
    udf_infos_(0, NULL, ObModIds::OB_SCHEMA_MODEL),
    udf_map_(ObModIds::OB_SCHEMA_MODEL)
  {
  }

ObPythonUDFMgr::ObPythonUDFMgr(common::ObIAllocator &allocator) :
    is_inited_(false),
    local_allocator_(ObModIds::OB_SCHEMA_GETTER_GUARD),
    allocator_(allocator),
    udf_infos_(0, NULL, ObModIds::OB_SCHEMA_MODEL),
    udf_map_(ObModIds::OB_SCHEMA_MODEL)
{
}

ObPythonUDFMgr::~ObPythonUDFMgr()
{
}

int ObPythonUDFMgr::init()
{
  int ret = OB_SUCCESS;
  if (is_inited_) {
    ret = OB_INIT_TWICE;
    LOG_WARN("init private python udf manager twice", K(ret));
  } else if (OB_FAIL(udf_map_.init())) {
    LOG_WARN("init private python udf map failed", K(ret));
  } else {
    is_inited_ = true;
  }
  return ret;
}

void ObPythonUDFMgr::reset()
{
  if (!is_inited_) {
    LOG_WARN_RET(OB_NOT_INIT, "udf manger not init");
  } else {
    udf_infos_.clear();
    udf_map_.clear();
  }
}

ObPythonUDFMgr &ObPythonUDFMgr::operator =(const ObPythonUDFMgr &other)
{
  int ret = OB_SUCCESS;
  if (!is_inited_) {
    ret = OB_NOT_INIT;
    LOG_WARN("python udf manager not init", K(ret));
  } else  if (OB_FAIL(assign(other))) {
    LOG_WARN("assign failed", K(ret));
  }
  return *this;
}

int ObPythonUDFMgr::assign(const ObPythonUDFMgr &other)
{
  int ret = OB_SUCCESS;
  if (!is_inited_) {
    ret = OB_NOT_INIT;
    LOG_WARN("python udf manager not init", K(ret));
  } else if (this != &other) {
    if (OB_FAIL(udf_map_.assign(other.udf_map_))) {
      LOG_WARN("assign udf map failed", K(ret));
    } else if (OB_FAIL(udf_infos_.assign(other.udf_infos_))) {
      LOG_WARN("assign udf infos vector failed", K(ret));
    }
  }
  return ret;
}

int ObPythonUDFMgr::deep_copy(const ObPythonUDFMgr &other)
{
  int ret = OB_SUCCESS;
  UNUSED(other);
  if (!is_inited_) {
    ret = OB_NOT_INIT;
    LOG_WARN("python udf manager not init", K(ret));
  } else if (this != &other) {
    reset();
    for (UDFIter iter = other.udf_infos_.begin();
       OB_SUCC(ret) && iter != other.udf_infos_.end(); iter++) {
      ObSimpleModelSchema *udf_info = *iter;
      if (OB_ISNULL(udf_info)) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("NULL ptr", K(udf_info), K(ret));
      } else if (OB_FAIL(add_model(*udf_info))) {
        LOG_WARN("add outline failed", K(*udf_info), K(ret));
      }
    }
  }
  return ret;
}

int ObPythonUDFMgr::get_model_schema_with_name(const uint64_t tenant_id,
                                               const common::ObString &name,
                                               const ObSimpleModelSchema *&model_schema) const
{
  int ret = OB_SUCCESS;
  LOG_WARN("get into ObPythonUDFMgr::check_model_exist_with_name", KR(ret));
  model_schema = NULL;
  if (!is_inited_) {
    ret = OB_NOT_INIT;
    LOG_WARN("not init", K(ret));
  } else if (OB_INVALID_ID == tenant_id
             || name.empty()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("invalid argument", K(ret), K(tenant_id), K(name));
  } else {
    ObSimpleModelSchema *tmp_schema = NULL;
    ObPythonUDFHashWrapper hash_wrap(tenant_id, name);
    if (OB_FAIL(udf_map_.get_refactored(hash_wrap, tmp_schema))) {
      if (OB_HASH_NOT_EXIST == ret) {
        ret = OB_SUCCESS;
        LOG_DEBUG("python udf is not exist", K(tenant_id), K(name));
      }
    } else {
      model_schema = tmp_schema;
    }
  }
  //判断model_schema是否为空
  if (model_schema == NULL) {
    LOG_WARN("model schema is NULL", KR(ret));
  }
  return ret;
}

int ObPythonUDFMgr::add_model(const ObSimpleModelSchema &udf_schema)
{
  int ret = OB_SUCCESS;
  LOG_WARN("get into add_model", K(ret));
  int overwrite = 1;
  ObSimpleModelSchema *new_udf_schema = NULL;
  UDFIter iter = NULL;
  ObSimpleModelSchema *replaced_udf = NULL;
  if (!is_inited_) {
    ret = OB_NOT_INIT;
    LOG_WARN("python udf manager not init", K(ret));
  } else if (OB_UNLIKELY(!udf_schema.is_valid())) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("invalid argument", K(ret), K(udf_schema));
  } else if (OB_FAIL(ObSchemaUtils::alloc_schema(allocator_, udf_schema, new_udf_schema))) {
    LOG_WARN("alloca udf schema failed", K(ret));
  } else if (OB_ISNULL(new_udf_schema)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("NULL ptr", K(new_udf_schema), K(ret));
  } else if (OB_FAIL(udf_infos_.replace(new_udf_schema,
                                        iter,
                                        compare_model,
                                        equal_model,
                                        replaced_udf))) {
      LOG_WARN("failed to add python udf schema", K(ret));
  } else {
    ObPythonUDFHashWrapper hash_wrapper(new_udf_schema->get_tenant_id(),
                                        new_udf_schema->get_model_name());
    if (OB_FAIL(udf_map_.set_refactored(hash_wrapper, new_udf_schema, overwrite))) {
      LOG_WARN("build python udf hash map failed", K(ret));
    }
  }
  if (udf_infos_.count() != udf_map_.item_count()) {
    LOG_WARN("python udf info is non-consistent",
             "python udf infos count", udf_infos_.count(),
             "python udf map item count", udf_map_.item_count(),
             "model name", udf_schema.get_model_name());
    int tmp_ret = OB_SUCCESS;
    if (OB_SUCCESS != (tmp_ret = ObPythonUDFMgr::rebuild_model_hashmap(udf_infos_, udf_map_))) {
      LOG_WARN("rebuild python udf hashmap failed", K(tmp_ret));
    }
  }
  return ret;
}

int ObPythonUDFMgr::rebuild_model_hashmap(const UDFInfos &udf_infos, ObUDFMap& udf_map)
{
  int ret = OB_SUCCESS;
  udf_map.clear();
  ConstUDFIter iter = udf_infos.begin();
  for (; iter != udf_infos.end() && OB_SUCC(ret); ++iter) {
    ObSimpleModelSchema *udf_schema = *iter;
    if (OB_ISNULL(udf_schema)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("udf schema is NULL", K(udf_schema), K(ret));
    } else {
      bool overwrite = true;
      ObPythonUDFHashWrapper hash_wrapper(udf_schema->get_tenant_id(),
                                          udf_schema->get_model_name());
      if (OB_FAIL(udf_map.set_refactored(hash_wrapper, udf_schema, overwrite))) {
        LOG_WARN("build udf hash map failed", K(ret));
      }
    }
  }
  return ret;
}

int ObPythonUDFMgr::del_schemas_in_tenant(const uint64_t tenant_id)
{
  int ret = OB_SUCCESS;
  if (OB_INVALID_ID == tenant_id) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("invalid argument", K(ret), K(tenant_id));
  } else {
    ObArray<const ObSimpleModelSchema *> schemas;
    if (OB_FAIL(get_model_schemas_in_tenant(tenant_id, schemas))) {
      LOG_WARN("get model schemas failed", K(ret), K(tenant_id));
    } else {
      FOREACH_CNT_X(schema, schemas, OB_SUCC(ret)) {
        ObTenantModelId tenant_model_id(tenant_id,
                                    (*schema)->get_model_name());
        if (OB_FAIL(del_model(tenant_model_id))) {
          LOG_WARN("del python udf failed",
                   "tenant_id", tenant_model_id.tenant_id_,
                   "model_name", tenant_model_id.model_name_,
                   K(ret));
        }
      }
    }
  }
  return ret;
}


int ObPythonUDFMgr::add_models(const common::ObIArray<ObSimpleModelSchema> &udf_schemas)
{
  int ret = OB_SUCCESS;
  for (int64_t i = 0; i < udf_schemas.count() && OB_SUCC(ret); ++i) {
    if (OB_FAIL(add_model(udf_schemas.at(i)))) {
      LOG_WARN("push python udf failed", K(ret));
    }
  }
  return ret;
}

int ObPythonUDFMgr::del_model(const ObTenantModelId &udf)
{
  int ret = OB_SUCCESS;
  int hash_ret = OB_SUCCESS;
  ObSimpleModelSchema *schema_to_del = NULL;
  if (!udf.is_valid()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("invalid argument", K(ret), K(udf));
  } else if (OB_FAIL(udf_infos_.remove_if(udf,
                                          compare_with_tenant_model_id,
                                          equal_to_tenant_model_id,
                                          schema_to_del))) {
    LOG_WARN("failed to remove python udf schema, ",
             "tenant id", udf.tenant_id_,
             "udf name", udf.model_name_,
             K(ret));
  } else if (OB_ISNULL(schema_to_del)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("removed udf schema return NULL, ",
             "tenant id", udf.tenant_id_,
             "udf name", udf.model_name_,
             K(ret));
  } else {
    ObPythonUDFHashWrapper udf_wrapper(schema_to_del->get_tenant_id(),
                                       schema_to_del->get_model_name());
    hash_ret = udf_map_.erase_refactored(udf_wrapper);
    if (OB_SUCCESS != hash_ret) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("failed delete udf from udf hashmap, ",
               K(ret),
               K(hash_ret),
               "tenant_id", schema_to_del->get_tenant_id(),
               "name", schema_to_del->get_model_name());
    }
  }
  if (udf_infos_.count() != udf_map_.item_count()) {
    LOG_WARN("python udf info is non-consistent",
             "python udf infos count", udf_infos_.count(),
             "python udf map item count", udf_map_.item_count(),
             "python udf name", udf.model_name_);
    int tmp_ret = OB_SUCCESS;
    if (OB_SUCCESS != (tmp_ret = ObPythonUDFMgr::rebuild_model_hashmap(udf_infos_, udf_map_))) {
      LOG_WARN("rebuild udf hashmap failed", K(tmp_ret));
    }
  }
  return ret;
}

bool ObPythonUDFMgr::compare_with_tenant_model_id(const ObSimpleModelSchema *lhs,
                                                  const ObTenantModelId &tenant_model_id)
{
  return NULL != lhs ? (lhs->get_tenant_model_id() < tenant_model_id) : false;
}

bool ObPythonUDFMgr::equal_to_tenant_model_id(const ObSimpleModelSchema *lhs,
                                              const ObTenantModelId &tenant_model_id)
{
  return NULL != lhs ? (lhs->get_tenant_model_id() == tenant_model_id) : false;
}


int ObPythonUDFMgr::get_model_schemas_in_tenant(const uint64_t tenant_id,
    ObIArray<const ObSimpleModelSchema *> &udf_schemas) const
{
  int ret = OB_SUCCESS;
  udf_schemas.reset();

  ConstUDFIter tenant_udf_begin = udf_infos_.begin();
  for (ConstUDFIter iter = tenant_udf_begin;
      OB_SUCC(ret) && iter != udf_infos_.end(); ++iter) {
    const ObSimpleModelSchema *udf = NULL;
    if (OB_ISNULL(udf = *iter)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("NULL ptr", K(ret), K(udf));
    } else if (tenant_id != udf->get_tenant_id()) {
      //do nothing
    } else if (OB_FAIL(udf_schemas.push_back(udf))) {
      LOG_WARN("push back udf failed", K(ret));
    }
  }

  return ret;
}

int ObPythonUDFMgr::get_model_schema(const uint64_t model_id,
                                   const ObSimpleModelSchema *&udf_schema) const
{
  int ret = OB_SUCCESS;
  // traverse is not a good idea, but it seems that i have no choice.
  for (int64_t i = 0; i < udf_infos_.count() && OB_SUCC(ret); ++i) {
    if (model_id == udf_infos_.at(i)->get_model_id()) {
      udf_schema = udf_infos_.at(i);
    }
  }
  return ret;
}

int ObPythonUDFMgr::get_model_schema_count(int64_t &udf_schema_count) const
{
  int ret = OB_SUCCESS;
  if (!is_inited_) {
    ret = OB_NOT_INIT;
    LOG_WARN("udf manager not init", K(ret));
  } else {
    udf_schema_count = udf_infos_.size();
  }
  return ret;
}

int ObPythonUDFMgr::get_schema_statistics(ObSchemaStatisticsInfo &schema_info) const
{
  int ret = OB_SUCCESS;
  schema_info.reset();
  schema_info.schema_type_ = UDF_SCHEMA;
  if (!is_inited_) {
    ret = OB_NOT_INIT;
    LOG_WARN("not init", K(ret));
  } else {
    schema_info.count_ = udf_infos_.size();
    for (ConstUDFIter it = udf_infos_.begin(); OB_SUCC(ret) && it != udf_infos_.end(); it++) {
      if (OB_ISNULL(*it)) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("schema is null", K(ret));
      } else {
        schema_info.size_ += (*it)->get_convert_size();
      }
    }
  }
  return ret;
}

}// end schema
}// end share
}// end oceanbase

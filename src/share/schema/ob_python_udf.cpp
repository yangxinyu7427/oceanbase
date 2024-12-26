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
#include "ob_python_udf.h"
#include <sstream>

namespace oceanbase
{

using namespace std;
using namespace common;

namespace share
{
namespace schema
{

ObPythonUDF::ObPythonUDF(common::ObIAllocator *allocator)
    : ObSchema(allocator), tenant_id_(common::OB_INVALID_ID), udf_id_(common::OB_INVALID_ID), name_(), arg_num_(0), arg_names_(), arg_types_(),
      ret_(ObPythonUdfEnumType::PyUdfRetType::UDF_UNINITIAL), pycall_(), schema_version_(common::OB_INVALID_VERSION),
      isModelSpecific_(false), model_num_(0), udf_model_names_(), udf_model_()
{
  reset();
}

ObPythonUDF::ObPythonUDF(const ObPythonUDF &src_schema)
    : ObSchema(), tenant_id_(common::OB_INVALID_ID), udf_id_(common::OB_INVALID_ID), name_(), arg_num_(0), arg_names_(), arg_types_(),
      ret_(ObPythonUdfEnumType::PyUdfRetType::UDF_UNINITIAL), pycall_(), schema_version_(common::OB_INVALID_VERSION),
      isModelSpecific_(false), model_num_(0), udf_model_names_(), udf_model_()
{
  reset();
  *this = src_schema;
}

ObPythonUDF::~ObPythonUDF()
{
}

int ObPythonUDF::get_arg_names_arr(common::ObIAllocator &allocator,
                                   common::ObSEArray<common::ObString, 16> &udf_attributes_names) const {
  int ret = OB_SUCCESS;
  udf_attributes_names.reuse();
  char* arg_names_str = const_cast<char*>(get_arg_names());
  std::istringstream ss(arg_names_str);
  std::string token;
  while (std::getline(ss, token, ',')) {
      char *ptr = static_cast<char *>(allocator.alloc(token.length()));
      MEMCPY(ptr, token.c_str(), token.length());
      udf_attributes_names.push_back(common::ObString(token.length(), ptr));
  }
  if(udf_attributes_names.count() != arg_num_) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("fail to resolve arg names string", K(ret));
  }
  return ret;
}

int ObPythonUDF::get_arg_types_arr(common::ObSEArray<ObPythonUdfEnumType::PyUdfRetType, 16> &udf_attributes_types) const {
  int ret = OB_SUCCESS;
  udf_attributes_types.reuse();
  char* arg_types_str = const_cast<char*>(get_arg_types());
  std::istringstream ss(arg_types_str);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token == "STRING") {
      udf_attributes_types.push_back(ObPythonUdfEnumType::PyUdfRetType::STRING);
    } else if (token == "INTEGER") {
      udf_attributes_types.push_back(ObPythonUdfEnumType::PyUdfRetType::INTEGER);
    } else if (token == "REAL") {
      udf_attributes_types.push_back(ObPythonUdfEnumType::PyUdfRetType::REAL);
    } else if (token == "DECIMAL") {
      udf_attributes_types.push_back(ObPythonUdfEnumType::PyUdfRetType::DECIMAL);
    }
  }
  if(udf_attributes_types.count() != arg_num_) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("fail to resolve arg types string", K(ret));
  }
  return ret;
}

int ObPythonUDF::get_udf_model_names_arr(common::ObIAllocator &allocator,
                                         common::ObSEArray<common::ObString, 16> &udf_model_names) const {
  int ret = OB_SUCCESS;
  udf_model_names.reuse();
  char* udf_model_names_str = const_cast<char*>(get_model_names());
  std::istringstream ss(udf_model_names_str);
  std::string token;
  while (std::getline(ss, token, ',')) {
      char *ptr = static_cast<char *>(allocator.alloc(token.length()));
      MEMCPY(ptr, token.c_str(), token.length());
      udf_model_names.push_back(common::ObString(token.length(), ptr));
  }
  if(udf_model_names.count() != model_num_) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("fail to resolve udf model names string", K(ret));
  }
  return ret;
}

int ObPythonUDF::insert_udf_model_info(ObUdfModel &model_info) {
  int ret = OB_SUCCESS;
  if (OB_FAIL(udf_model_.push_back(model_info))) {
    LOG_WARN("fail to insert udf model info", K(ret)); 
  }
  return ret;
}

int ObPythonUDF::check_pycall() const {
  int ret = OB_SUCCESS;
  const char* python_code = get_pycall();
  // Py_Initialize();
  // if (PyRun_SimpleString(python_code) != 0) {
  //   ret = OB_INVALID_ARGUMENT;
  //   LOG_WARN("python code raise an exception", K(ret));
  // }
  if (strstr(python_code, "pyinitial") == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("pycall lack pyinitial", K(ret));
  } else if (strstr(python_code, "pyfun") == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("pycall lack pyfun", K(ret));
  }
  // Py_FinalizeEx();
  return ret;
}

ObPythonUDF& ObPythonUDF::operator= (const ObPythonUDF &other) {
  if (this != &other) {
    reset();
    int ret = OB_SUCCESS;
    error_ret_ = other.error_ret_;
    tenant_id_ = other.tenant_id_;
    udf_id_ = other.udf_id_;
    arg_num_ = other.arg_num_;
    schema_version_ = other.schema_version_;
    ret_ = other.ret_;
    model_num_ = other.model_num_;
    isModelSpecific_ = other.isModelSpecific_;
    if (OB_FAIL(deep_copy_str(other.name_, name_))) {
      LOG_WARN("Fail to deep copy name", K(ret));
    } else if (OB_FAIL(deep_copy_str(other.arg_names_, arg_names_))) {
      LOG_WARN("Fail to deep copy arg names", K(ret));
    } else if (OB_FAIL(deep_copy_str(other.arg_types_, arg_types_))) {
      LOG_WARN("Fail to deep copy arg types", K(ret));
    } else if (OB_FAIL(deep_copy_str(other.pycall_, pycall_))) {
      LOG_WARN("Fail to deep copy pycall", K(ret));
    } else if (OB_FAIL(deep_copy_str(other.udf_model_names_, udf_model_names_))) {
      LOG_WARN("Fail to deep copy udf model names", K(ret));
    }
    LOG_WARN("get into ObPythonUDF::operator=", K(ret), K(model_num_));
    if (OB_FAIL(ret)) {
      error_ret_ = ret;
    }
  }
  return *this;
}

void ObPythonUDF::reset()
{
  tenant_id_ = OB_INVALID_ID;
  udf_id_ = OB_INVALID_ID;
  name_.reset();
  arg_names_.reset();
  arg_num_ = 0;
  arg_types_.reset();
  ret_ = ObPythonUdfEnumType::PyUdfRetType::UDF_UNINITIAL;
  pycall_.reset();
  isModelSpecific_ = false;
  model_num_ = 0;
  udf_model_names_.reset();
  udf_model_.reset();
  ObSchema::reset();
}

OB_SERIALIZE_MEMBER(ObPythonUDF,
				            tenant_id_,
                    udf_id_,
                    name_,
                    arg_num_,
                    arg_names_,
                    arg_types_,
				            ret_,
                    pycall_,
                    model_num_,
                    isModelSpecific_,
                    udf_model_names_,
                    udf_model_);

OB_SERIALIZE_MEMBER(ObPythonUDFMeta,
                    name_,
                    ret_,
                    pycall_,
                    udf_attributes_names_,
                    udf_attributes_types_,
                    init_,
                    batch_size_,
                    batch_size_const_,
                    model_type_,
                    udf_model_meta_);


/////////////////////////////////////////////

ObUdfModel::ObUdfModel(common::ObIAllocator *allocator)
    : ObSchema(), tenant_id_(common::OB_INVALID_ID), model_id_(common::OB_INVALID_ID), model_name_(), model_type_(), 
      framework_(), model_path_(), arg_num_(0), arg_names_(), arg_types_(), ret_(ObPythonUdfEnumType::PyUdfRetType::UDF_UNINITIAL), 
      schema_version_(common::OB_INVALID_VERSION) 
{
  reset();
}

ObUdfModel::ObUdfModel(const ObUdfModel &src_schema)
    : ObSchema(), tenant_id_(common::OB_INVALID_ID), model_id_(common::OB_INVALID_ID), model_name_(), model_type_(), 
      framework_(), model_path_(), arg_num_(0), arg_names_(), arg_types_(), ret_(ObPythonUdfEnumType::PyUdfRetType::UDF_UNINITIAL),
      schema_version_(common::OB_INVALID_VERSION) 
{
  reset();
  *this = src_schema;
}

ObUdfModel::~ObUdfModel()
{
}

ObUdfModel& ObUdfModel::operator=(const ObUdfModel &other) {
  if (this != &other) {
    reset();
    int ret = OB_SUCCESS;
    error_ret_ = other.error_ret_;
    tenant_id_ = other.tenant_id_;
    model_id_ = other.model_id_;
    framework_ = other.framework_;
    model_type_ = other.model_type_;
    arg_num_ = other.arg_num_;
    ret_ = other.ret_;
    schema_version_ = other.schema_version_;
    if (OB_FAIL(deep_copy_str(other.model_name_, model_name_))) {
      LOG_WARN("Fail to deep copy model name", K(ret));
    } else if (OB_FAIL(deep_copy_str(other.model_path_, model_path_))) {
      LOG_WARN("Fail to deep copy model path", K(ret));
    } else if (OB_FAIL(deep_copy_str(other.arg_names_, arg_names_))) {
      LOG_WARN("Fail to deep copy arg names", K(ret));
    } else if (OB_FAIL(deep_copy_str(other.arg_types_, arg_types_))) {
      LOG_WARN("Fail to deep copy arg types", K(ret));
    } 
    if (OB_FAIL(ret)) {
      error_ret_ = ret;
    }
  }
  return *this;
}

void ObUdfModel::reset()
{
  tenant_id_ = OB_INVALID_ID;
  model_id_ = OB_INVALID_ID;
  model_name_.reset();
  framework_ = ObPythonUdfEnumType::ModelFrameworkType::INVALID_FRAMEWORK_TYPE;
  model_type_ = ObPythonUdfEnumType::ModelType::INVALID_MODEL_TYPE;
  model_path_.reset();  
  arg_names_.reset();
  arg_types_.reset();
  ret_ = ObPythonUdfEnumType::PyUdfRetType::UDF_UNINITIAL;
  ObSchema::reset();
}

OB_SERIALIZE_MEMBER(ObUdfModel,
				            tenant_id_,
                    model_id_,
                    model_name_,
                    model_type_,
                    framework_,
                    model_path_,
                    arg_num_,
                    arg_names_,
                    arg_types_,
				            ret_,
                    schema_version_);

OB_SERIALIZE_MEMBER(ObUdfModelMeta,
                    model_name_,
                    framework_,
                    model_type_,
                    model_path_,
                    model_attributes_names_,
                    model_attributes_types_,
                    ret_);




}// end schema
}// end share
}// end oceanbase
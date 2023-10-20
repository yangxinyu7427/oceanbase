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
    : ObSchema(allocator), tenant_id_(common::OB_INVALID_ID), model_id_(common::OB_INVALID_ID), name_(), arg_num_(0), arg_names_(), arg_types_(),
      ret_(PyUdfRetType::UDF_UNINITIAL), pycall_(), schema_version_(common::OB_INVALID_VERSION)
{
  reset();
}

ObPythonUDF::ObPythonUDF(const ObPythonUDF &src_schema)
    : ObSchema(), tenant_id_(common::OB_INVALID_ID), model_id_(common::OB_INVALID_ID), name_(), arg_num_(0), arg_names_(), arg_types_(),
      ret_(PyUdfRetType::UDF_UNINITIAL), pycall_(), schema_version_(common::OB_INVALID_VERSION)
{
  reset();
  *this = src_schema;
}

ObPythonUDF::~ObPythonUDF()
{
}

void ObPythonUDF::set_arg_names(const char* arg_name) {
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

void ObPythonUDF::set_arg_types(const std::string type_string) {
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

common::ObSEArray<ObPythonUDF::PyUdfRetType, 16> ObPythonUDF::get_arg_types_arr() {
    common::ObSEArray<ObPythonUDF::PyUdfRetType, 16> udf_attributes_types;
    char* arg_types_str = const_cast<char*>(get_arg_types());
    std::istringstream ss(arg_types_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token == "STRING") {
            udf_attributes_types.push_back(ObPythonUDF::PyUdfRetType::STRING);
        } else if (token == "INTEGER") {
            udf_attributes_types.push_back(ObPythonUDF::PyUdfRetType::INTEGER);
        } else if (token == "REAL") {
            udf_attributes_types.push_back(ObPythonUDF::PyUdfRetType::REAL);
        } else if (token == "DECIMAL") {
            udf_attributes_types.push_back(ObPythonUDF::PyUdfRetType::DECIMAL);
        }
      }
    return udf_attributes_types;
} 

void ObPythonUDF::reset()
{
  tenant_id_ = OB_INVALID_ID;
  name_.reset();
  arg_num_ = 0;
  arg_names_.reset();
  arg_types_.reset();
  ret_ = PyUdfRetType::UDF_UNINITIAL;
  pycall_.reset();
  ObSchema::reset();
}

OB_SERIALIZE_MEMBER(ObPythonUDF,
					          tenant_id_,
                    model_id_,
                    name_,
                    arg_num_,
                    arg_names_,
                    arg_types_,
					          ret_,
                    pycall_);

OB_SERIALIZE_MEMBER(ObPythonUDFMeta,
                    ret_,
                    pycall_,
                    udf_attributes_types_);

}// end schema
}// end share
}// end oceanbase
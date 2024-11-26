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
#include "ob_udf_model.h"
#include <sstream>

namespace oceanbase
{

using namespace std;
using namespace common;

namespace share
{
namespace schema
{

ObUdfModel::ObUdfModel(common::ObIAllocator *allocator)
    : ObSchema(), tenant_id_(common::OB_INVALID_ID), model_id_(common::OB_INVALID_ID), model_name_(), model_type_(), framework_(), model_path_(), schema_version_(common::OB_INVALID_VERSION) 
{
  reset();
}

ObUdfModel::ObUdfModel(const ObUdfModel &src_schema)
    : ObSchema(), tenant_id_(common::OB_INVALID_ID), model_id_(common::OB_INVALID_ID), model_name_(), model_type_(), framework_(), model_path_(), schema_version_(common::OB_INVALID_VERSION) 
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
    schema_version_ = other.schema_version_;
    if (OB_FAIL(deep_copy_str(other.model_name_, model_name_))) {
      LOG_WARN("Fail to deep copy model name", K(ret));
    } else if (OB_FAIL(deep_copy_str(other.model_path_, model_path_))) {
      LOG_WARN("Fail to deep copy model path", K(ret));
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
  framework_ = ModelFrameworkType::INVALID_FRAMEWORK_TYPE;
  model_type_ = ModelType::INVALID_MODEL_TYPE;
  model_path_.reset();
  ObSchema::reset();
}

OB_SERIALIZE_MEMBER(ObUdfModel,
				            tenant_id_,
                    model_id_,
                    model_name_,
                    model_type_,
                    framework_,
                    model_path_,
                    schema_version_);

OB_SERIALIZE_MEMBER(ObUdfModelMeta,
                    model_name_,
                    framework_,
                    model_type_,
                    model_path_);

}// end schema
}// end share
}// end oceanbase
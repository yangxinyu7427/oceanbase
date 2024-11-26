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

#ifndef _OB_UDF_MODEL_H
#define _OB_UDF_MODEL_H 1
#include "lib/ob_define.h"
#include "lib/string/ob_string.h"
#include "lib/container/ob_array.h"
#include "ob_schema_struct.h"
// #include <Python.h>

namespace oceanbase
{
namespace share
{
namespace schema
{

class ObTableSchema;
class ObUdfModel : public ObSchema
{
    OB_UNIS_VERSION_V(1);

public:
    //枚举计算结果后返回值类型
    enum ModelFrameworkType {
        INVALID_FRAMEWORK_TYPE,
        ONNX,
        SKLEARN,
        PYTORCH,
        UNSUPPORTED
    };

    enum ModelType {
        INVALID_MODEL_TYPE,
        DECISION_TREE
    };
    
public:
    ObUdfModel() : ObSchema(), tenant_id_(common::OB_INVALID_ID), model_id_(common::OB_INVALID_ID), 
                   model_name_(), model_type_(ModelType::INVALID_MODEL_TYPE), framework_(ModelFrameworkType::INVALID_FRAMEWORK_TYPE), 
                   model_path_(), schema_version_(common::OB_INVALID_VERSION) 
                    { reset(); };
    explicit ObUdfModel(common::ObIAllocator *allocator);
    ObUdfModel(const ObUdfModel &src_schema);
    virtual ~ObUdfModel();

    //operators
    ObUdfModel& operator=(const ObUdfModel &src_schema);
    // bool operator==(const ObUdfModel &r) const;
    // bool operator!=(const ObUdfModel &r) const;

    //set methods
    inline void set_tenant_id(const uint64_t id) { tenant_id_ = id; }
    inline void set_model_id(const uint64_t id) { model_id_ = id; }
    inline int set_model_name(const common::ObString &model_name) { return deep_copy_str(model_name, model_name_); }
    inline void set_model_type(const enum ModelType model_type) { model_type_ = model_type; }
    inline void set_model_type(const int model_type) { model_type_ = ModelType(model_type); }
    inline void set_framework(const enum ModelFrameworkType framework_type) { framework_ = framework_type; }
    inline void set_framework(const int framework_type) { framework_ = ModelFrameworkType(framework_type); }
    inline int set_model_path(const common::ObString &model_path) { return deep_copy_str(model_path, model_path_); }
    inline void set_schema_version(int64_t version) { schema_version_ = version; }

    //get methods
    inline uint64_t get_tenant_id() const { return tenant_id_; }
    inline uint64_t get_model_id() const { return model_id_; }
    inline const char *get_model_name() const { return extract_str(model_name_); }
    inline const common::ObString &get_model_name_str() const { return model_name_; }
    inline enum ModelType get_model_type() const { return model_type_; }
    inline enum ModelFrameworkType get_framework() const { return framework_; }
    inline const char *get_model_path() const { return extract_str(model_path_); }
    inline const common::ObString &get_model_path_str() const { return model_path_; }  
    inline int64_t get_schema_version() const { return schema_version_; }

    //other
    virtual void reset() override;

    TO_STRING_KV(K_(tenant_id),
                 K_(model_id),
                 K_(model_name),
                 K_(model_type),
                 K_(framework),
                 K_(model_path),
                 K_(schema_version));

public:
    uint64_t tenant_id_;
    uint64_t model_id_;
    common::ObString model_name_;
    enum ModelType model_type_;   //model type(lr,rf)
    enum ModelFrameworkType framework_;    //model framework(onnx,sklearn)
    common::ObString model_path_;   //model path
    // common::ObString model_data_;   //model data
    int64_t schema_version_; //the last modify timestamp of this version
};

/////////////////////////////////////////////

class ObUdfModelMeta
{
  OB_UNIS_VERSION_V(1);
public :
  ObUdfModelMeta() : model_name_(), framework_(ObUdfModel::ModelFrameworkType::INVALID_FRAMEWORK_TYPE), 
                     model_type_(ObUdfModel::ModelType::INVALID_MODEL_TYPE), model_path_() {} 
  virtual ~ObUdfModelMeta() = default;

  void assign(const ObUdfModelMeta &other) { 
    model_name_ = other.model_name_;
    framework_ = other.framework_;
    model_type_ = other.model_type_;
    model_path_ = other.model_path_;
  }

  ObUdfModelMeta &operator=(const class ObUdfModelMeta &other) {
    model_name_ = other.model_name_;
    framework_ = other.framework_;
    model_type_ = other.model_type_;
    model_path_ = other.model_path_;
    return *this;
  }

  TO_STRING_KV(K_(model_name),
               K_(framework),
               K_(model_type),
               K_(model_path));

  common::ObString model_name_; //模型名称
  ObUdfModel::ModelFrameworkType framework_;  //模型框架
  ObUdfModel::ModelType model_type_; //模型类型
  common::ObString model_path_; //模型路径
};

}
}
}

#endif /* _OB_UDF_MODEL_H */
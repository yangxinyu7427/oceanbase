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

#ifndef _OB_PYTHON_UDF_H
#define _OB_PYTHON_UDF_H 1
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

class ObPythonUdfEnumType {
public:
  //枚举计算结果后返回值类型
  enum ModelFrameworkType {
    INVALID_FRAMEWORK_TYPE,
    ONNX,
    SKLEARN,
    PYTORCH,
    UNSUPPORTED,
  };

  enum ModelType {
    INVALID_MODEL_TYPE,
    DECISION_TREE,
  };

  //枚举计算结果后返回值类型
  enum PyUdfRetType {
    UDF_UNINITIAL,
    STRING,
    INTEGER,
    REAL,
    DECIMAL,
  };

  //枚举UDF携带模型信息的类型
  enum PyUdfUsingType {
    INVALID,
    MODEL_SPECIFIC,
    ARBITRARY_CODE,
    NONE,
  };
};

class ObUdfModel : public ObSchema
{
    OB_UNIS_VERSION_V(1);
public:
    ObUdfModel() : ObSchema(), tenant_id_(common::OB_INVALID_ID), model_id_(common::OB_INVALID_ID), 
                   model_name_(), model_type_(ObPythonUdfEnumType::ModelType::INVALID_MODEL_TYPE), framework_(ObPythonUdfEnumType::ModelFrameworkType::INVALID_FRAMEWORK_TYPE), 
                   model_path_(), arg_num_(0), arg_names_(), arg_types_(), ret_(ObPythonUdfEnumType::PyUdfRetType::UDF_UNINITIAL),
                   schema_version_(common::OB_INVALID_VERSION) 
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
    inline void set_model_type(const enum ObPythonUdfEnumType::ModelType model_type) { model_type_ = model_type; }
    inline void set_model_type(const int model_type) { model_type_ = ObPythonUdfEnumType::ModelType(model_type); }
    inline void set_framework(const enum ObPythonUdfEnumType::ModelFrameworkType framework_type) { framework_ = framework_type; }
    inline void set_framework(const int framework_type) { framework_ = ObPythonUdfEnumType::ModelFrameworkType(framework_type); }
    inline int set_model_path(const common::ObString &model_path) { return deep_copy_str(model_path, model_path_); }
    inline void set_ret(const enum ObPythonUdfEnumType::PyUdfRetType ret) { ret_ = ret; }
    inline void set_ret(const int ret) { ret_ = ObPythonUdfEnumType::PyUdfRetType(ret); }
    inline void set_arg_num(const int arg_num) { arg_num_ = arg_num; }
    inline int set_arg_names(const common::ObString arg_names) { return deep_copy_str(arg_names, arg_names_); }
    inline int set_arg_types(const common::ObString arg_types) { return deep_copy_str(arg_types, arg_types_); }
    inline void set_schema_version(int64_t version) { schema_version_ = version; }

    //get methods
    inline uint64_t get_tenant_id() const { return tenant_id_; }
    inline uint64_t get_model_id() const { return model_id_; }
    inline const char *get_model_name() const { return extract_str(model_name_); }
    inline const common::ObString &get_model_name_str() const { return model_name_; }
    inline enum ObPythonUdfEnumType::ModelType get_model_type() const { return model_type_; }
    inline enum ObPythonUdfEnumType::ModelFrameworkType get_framework() const { return framework_; }
    inline const char *get_model_path() const { return extract_str(model_path_); }
    inline const common::ObString &get_model_path_str() const { return model_path_; }
    inline int get_arg_num() const { return arg_num_; }
    const char *get_arg_names() const { return extract_str(arg_names_); }
    inline const common::ObString &get_arg_names_str() const { return arg_names_; }
    inline const char *get_arg_types() const { return extract_str(arg_types_); }
    inline const common::ObString &get_arg_types_str() const { return arg_types_; }
    inline enum ObPythonUdfEnumType::PyUdfRetType get_ret() const { return ret_; } 
    inline int64_t get_schema_version() const { return schema_version_; }

    int get_arg_names_arr(common::ObIAllocator &allocator, 
                          common::ObSEArray<common::ObString, 16> &model_attributes_names) const;
    int get_arg_types_arr(common::ObSEArray<ObPythonUdfEnumType::PyUdfRetType, 16> &model_attributes_types) const;

    //other
    virtual void reset() override;

    TO_STRING_KV(K_(tenant_id),
                 K_(model_id),
                 K_(model_name),
                 K_(model_type),
                 K_(framework),
                 K_(model_path),
                 K_(arg_num),
                 K_(arg_names),
                 K_(arg_types),
                 K_(ret),
                 K_(schema_version));

public:
    uint64_t tenant_id_;
    uint64_t model_id_;
    common::ObString model_name_;
    enum ObPythonUdfEnumType::ModelType model_type_;   //model type(lr,rf)
    enum ObPythonUdfEnumType::ModelFrameworkType framework_;    //model framework(onnx,sklearn)
    common::ObString model_path_;   //model path
    int arg_num_; //参数数量
    common::ObString arg_names_; //参数名称
    common::ObString arg_types_; //参数类型
    enum ObPythonUdfEnumType::PyUdfRetType ret_; //返回值类型
    // common::ObString model_data_;   //model data
    int64_t schema_version_; //the last modify timestamp of this version
};

/////////////////////////////////////////////

class ObUdfModelMeta
{
  OB_UNIS_VERSION_V(1);
public :
  ObUdfModelMeta() : model_name_(), framework_(ObPythonUdfEnumType::ModelFrameworkType::INVALID_FRAMEWORK_TYPE), 
                     model_type_(ObPythonUdfEnumType::ModelType::INVALID_MODEL_TYPE), model_path_(),
                     model_attributes_types_(),ret_(ObPythonUdfEnumType::PyUdfRetType::UDF_UNINITIAL) {} 
  virtual ~ObUdfModelMeta() = default;

  void assign(const ObUdfModelMeta &other) { 
    model_name_ = other.model_name_;
    framework_ = other.framework_;
    model_type_ = other.model_type_;
    model_path_ = other.model_path_;
    model_attributes_names_ = other.model_attributes_names_;
    model_attributes_types_ = other.model_attributes_types_;
    ret_ = other.ret_;
  }

  ObUdfModelMeta &operator=(const class ObUdfModelMeta &other) {
    model_name_ = other.model_name_;
    framework_ = other.framework_;
    model_type_ = other.model_type_;
    model_path_ = other.model_path_;
    model_attributes_names_ = other.model_attributes_names_;
    model_attributes_types_ = other.model_attributes_types_;
    ret_ = other.ret_;
    return *this;
  }

  TO_STRING_KV(K_(model_name),
               K_(framework),
               K_(model_type),
               K_(model_path),
               K_(model_attributes_names),
               K_(model_attributes_types),
               K_(ret));

  common::ObString model_name_; //模型名称
  ObPythonUdfEnumType::ModelFrameworkType framework_;  //模型框架
  ObPythonUdfEnumType::ModelType model_type_; //模型类型
  common::ObString model_path_; //模型路径
  common::ObSEArray<common::ObString, 16> model_attributes_names_; //参数名称
  common::ObSEArray<ObPythonUdfEnumType::PyUdfRetType, 16> model_attributes_types_; //参数类型
  ObPythonUdfEnumType::PyUdfRetType ret_; //返回值类型
};

/////////////////////////////////////////////

class ObPythonUDF : public ObSchema
{
    OB_UNIS_VERSION_V(1);
public:
    ObPythonUDF() : ObSchema(), tenant_id_(common::OB_INVALID_ID), udf_id_(common::OB_INVALID_ID), name_(), arg_num_(0), arg_names_(), 
                    arg_types_(), ret_(ObPythonUdfEnumType::PyUdfRetType::UDF_UNINITIAL), pycall_(), schema_version_(common::OB_INVALID_VERSION), 
                    isModelSpecific_(false), model_num_(0), udf_model_names_(), udf_model_()
                    { reset(); };
    explicit ObPythonUDF(common::ObIAllocator *allocator);
    ObPythonUDF(const ObPythonUDF &src_schema);
    virtual ~ObPythonUDF();

    //operators
    ObPythonUDF& operator=(const ObPythonUDF &src_schema);
    // bool operator==(const ObPythonUDF &r) const;
    // bool operator!=(const ObPythonUDF &r) const;

    //set methods
    inline void set_tenant_id(const uint64_t id) { tenant_id_ = id; }
    inline void set_udf_id(const uint64_t id) { udf_id_ = id; }
    inline int set_name(const common::ObString &name) { return deep_copy_str(name, name_); }
    inline void set_ret(const enum ObPythonUdfEnumType::PyUdfRetType ret) { ret_ = ret; }
    inline void set_ret(const int ret) { ret_ = ObPythonUdfEnumType::PyUdfRetType(ret); }
    inline void set_arg_num(const int arg_num) { arg_num_ = arg_num; }
    inline int set_arg_names(const common::ObString arg_names) { return deep_copy_str(arg_names, arg_names_); }
    inline int set_arg_types(const common::ObString arg_types) { return deep_copy_str(arg_types, arg_types_); }
    inline int set_pycall(const common::ObString &pycall) { return deep_copy_str(pycall, pycall_); }
    inline void set_schema_version(int64_t version) { schema_version_ = version; }
    inline void set_model_num(const int model_num) { model_num_ = model_num; }
    inline void set_isModelSpecific(bool isModelSpecific) { isModelSpecific_ = isModelSpecific; }
    inline int set_model_names(const common::ObString udf_model_names) { return deep_copy_str(udf_model_names, udf_model_names_); }

    //get methods
    inline uint64_t get_tenant_id() const { return tenant_id_; }
    inline uint64_t get_udf_id() const { return udf_id_; }
    inline const char *get_name() const { return extract_str(name_); }
    inline const common::ObString &get_name_str() const { return name_; }
    inline int get_arg_num() const { return arg_num_; }
    const char *get_arg_names() const { return extract_str(arg_names_); }
    inline const common::ObString &get_arg_names_str() const { return arg_names_; }
    inline const char *get_arg_types() const { return extract_str(arg_types_); }
    inline const common::ObString &get_arg_types_str() const { return arg_types_; }
    inline enum ObPythonUdfEnumType::PyUdfRetType get_ret() const { return ret_; }
    inline const char *get_pycall() const { return extract_str(pycall_); }
    inline const common::ObString &get_pycall_str() const { return pycall_; }
    inline int64_t get_schema_version() const { return schema_version_; }
    inline int get_model_num() const { return model_num_; }
    inline bool get_isModelSpecific() const { return isModelSpecific_; }
    inline const char *get_model_names() const { return extract_str(udf_model_names_); }
    inline const common::ObString &get_udf_model_names_str() const { return udf_model_names_; }

    //only for retrieve udf
    inline const char *get_udf_name() const { return extract_str(name_); }
    inline common::ObString get_udf_name_str() const { return name_; }
    
    //other
    virtual void reset() override;
    int get_arg_names_arr(common::ObIAllocator &allocator,
                          common::ObSEArray<common::ObString, 16> &udf_attributes_names) const;
    int get_arg_types_arr(common::ObSEArray<ObPythonUdfEnumType::PyUdfRetType, 16> &udf_attributes_types) const;
    int get_udf_model_names_arr(common::ObIAllocator &allocator, common::ObSEArray<common::ObString, 16> &udf_model_names) const;
    int insert_udf_model_info(ObUdfModel &model_info);
    int check_pycall() const;

    TO_STRING_KV(K_(tenant_id),
                 K_(udf_id),
                 K_(name),
                 K_(arg_num),
                 K_(arg_names),
                 K_(arg_types),
                 K_(ret),
                 K_(pycall),
                 K_(schema_version),
                 K_(isModelSpecific),
                 K_(model_num),
                 K_(udf_model_names),
                 K_(udf_model));

public:
    typedef  ObPythonUdfEnumType::PyUdfRetType PyUdfRetType;
    uint64_t tenant_id_;
    uint64_t udf_id_;
    common::ObString name_;
    int arg_num_;                                                     // 参数数量
    common::ObString arg_names_;                                      // 参数名称
    common::ObString arg_types_;                                      // 参数类型
    enum ObPythonUdfEnumType::PyUdfRetType ret_;                      // 返回值类型
    common::ObString pycall_;                                         // python code
    int64_t schema_version_;                                          // the last modify timestamp of this version
    bool isModelSpecific_;                                            // 是否为single model specific udf
    int model_num_;                                                   // udf所绑定模型个数
    common::ObString udf_model_names_;                                // udf所绑定的模型名称
    common::ObSEArray<ObUdfModel, 16> udf_model_;                   // udf所的模型信息
};

/////////////////////////////////////////////

class ObPythonUDFMeta
{
  OB_UNIS_VERSION_V(1);
public :


  ObPythonUDFMeta() : name_(), ret_(ObPythonUdfEnumType::PyUdfRetType::UDF_UNINITIAL), pycall_(), 
                      udf_attributes_names_(), udf_attributes_types_(), init_(false), 
                      batch_size_(256), batch_size_const_(false), 
                      model_type_(ObPythonUdfEnumType::PyUdfUsingType::INVALID), udf_model_meta_(),is_retree_opt_(false),
                      ismerged_(false), merged_udf_names_(), origin_input_count_(0), 
                      has_new_output_model_path_(false), has_new_input_model_path_(false) {} 
  virtual ~ObPythonUDFMeta() = default;

  void assign(const ObPythonUDFMeta &other) { 
    name_ = other.name_;
    ret_ = other.ret_;
    pycall_ = other.pycall_;
    udf_attributes_names_ = other.udf_attributes_names_;
    udf_attributes_types_ = other.udf_attributes_types_;
    init_ = other.init_;
    batch_size_ = other.batch_size_;
    batch_size_const_ = other.batch_size_const_;
    model_type_ = other.model_type_;
    udf_model_meta_ = other.udf_model_meta_;
    ismerged_=other.ismerged_;
    merged_udf_names_=other.merged_udf_names_;
    origin_input_count_=other.origin_input_count_;
    has_new_output_model_path_=other.has_new_output_model_path_;
    new_output_model_path_=other.new_output_model_path_;
    has_new_input_model_path_=other.has_new_input_model_path_;
    new_input_model_path_=other.new_input_model_path_;
    can_be_used_model_path_=other.can_be_used_model_path_;
    opted_model_path_=other.opted_model_path_;
    model_path_=other.model_path_;
    is_retree_opt_ = other.is_retree_opt_;
  }

  ObPythonUDFMeta &operator=(const class ObPythonUDFMeta &other) {
    name_ = other.name_;
    ret_ = other.ret_;
    pycall_ = other.pycall_;
    udf_attributes_names_ = other.udf_attributes_names_;
    udf_attributes_types_ = other.udf_attributes_types_;
    init_ = other.init_;
    batch_size_ = other.batch_size_;
    batch_size_const_ = other.batch_size_const_;
    ismerged_=other.ismerged_;
    merged_udf_names_=other.merged_udf_names_;
    origin_input_count_=other.origin_input_count_;
    has_new_output_model_path_=other.has_new_output_model_path_;
    new_output_model_path_=other.new_output_model_path_;
    has_new_input_model_path_=other.has_new_input_model_path_;
    new_input_model_path_=other.new_input_model_path_;
    can_be_used_model_path_=other.can_be_used_model_path_;
    opted_model_path_=other.opted_model_path_;
    model_path_=other.model_path_;
    model_type_ = other.model_type_;
    udf_model_meta_ = other.udf_model_meta_;
    is_retree_opt_ = other.is_retree_opt_;
    return *this;
  }

  TO_STRING_KV(K_(name),
               K_(ret),
               K_(pycall),
               K_(udf_attributes_names),
               K_(udf_attributes_types),
               K_(init),
               K_(batch_size),
               K_(batch_size_const),
               K_(model_type),
               K_(udf_model_meta));

  common::ObString name_;                                                   // 函数名
  ObPythonUdfEnumType::PyUdfRetType ret_;                                           // 返回值类型
  common::ObString pycall_;                                                 // python code
  common::ObSEArray<common::ObString, 16> udf_attributes_names_;            // 参数名称
  common::ObSEArray<ObPythonUdfEnumType::PyUdfRetType, 16> udf_attributes_types_;   // 参数类型
  bool init_;                                                               // 是否已初始化
  int batch_size_;                                                          // 推理批次大小
  bool batch_size_const_;                                                   // batch_size是否动态调整
  ObPythonUdfEnumType::PyUdfUsingType model_type_;                                  // 是否包含模型信息
  common::ObSEArray<ObUdfModelMeta, 16> udf_model_meta_;                    // udf所使用的模型信息

  // dtpo
  bool is_retree_opt_;

  // yxy
  bool ismerged_;//是否经查询内冗余消除融合
  std::string opted_model_path_;//经查询内冗余消除策略优化后的模型地址
  common::ObSEArray<common::ObString, 16> merged_udf_names_; //融合的udf名
  int origin_input_count_; //初始udf的input数
  bool has_new_output_model_path_;
  std::string new_output_model_path_;//执行的过程中可以导出要缓存的中间结果的模型地址
  bool has_new_input_model_path_;
  std::string new_input_model_path_;//执行的过程中可以使用已缓存的中间结果的模型地址
  common::ObString can_be_used_model_path_;//检测出的可复用的历史模型地址
  common::ObString model_path_;//本udf对应的模型地址
};

}
}
}

#endif /* _OB_PYTHON_UDF_H */
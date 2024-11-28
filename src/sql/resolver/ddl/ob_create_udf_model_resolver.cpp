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
#define USING_LOG_PREFIX SQL_RESV
#include "sql/resolver/ddl/ob_create_udf_model_resolver.h"
namespace oceanbase
{
namespace sql
{
ObCreateUdfModelResolver::ObCreateUdfModelResolver(ObResolverParams &params)
    : ObDDLResolver(params)
{
}
ObCreateUdfModelResolver::~ObCreateUdfModelResolver()
{
}
int ObCreateUdfModelResolver::resolve(const ParseNode &parse_tree)
{
    int ret = OB_SUCCESS;
    ParseNode *create_udf_model_node = const_cast<ParseNode*>(&parse_tree);
    if (OB_ISNULL(create_udf_model_node)
        || T_CREATE_UDF_MODEL != create_udf_model_node->type_
        || 1 > create_udf_model_node->num_child_         //语法树根节点的孩子数不正确
        || OB_ISNULL(create_udf_model_node->children_)) {
      ret = OB_INVALID_ARGUMENT;
      SQL_RESV_LOG(WARN, "invalid argument.", K(ret));
    } else {
      ObCreateUdfModelStmt *create_udf_model_stmt = NULL;
      ObString model_name;
      if (OB_ISNULL(create_udf_model_stmt = create_stmt<ObCreateUdfModelStmt>())) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        SQL_RESV_LOG(ERROR, "failed to create create_udf_model_stmt", K(ret));
      } else {
        stmt_ = create_udf_model_stmt;
        obrpc::ObCreateUdfModelArg &create_udf_model_arg = create_udf_model_stmt->get_create_udf_model_arg();
        //get opt_if_exists ()
        //get model name
        ParseNode *relation_node = create_udf_model_node->children_[1];
        model_name = ObString(relation_node->str_len_, relation_node->str_value_);
        //set model name
        create_udf_model_arg.udf_model_.set_model_name(model_name);
        //resolve model param
        ParseNode *function_element_list_node = create_udf_model_node->children_[2];
        //set arg_num
        int arg_num = function_element_list_node->num_child_;
        create_udf_model_arg.udf_model_.set_arg_num(arg_num);
        std::string arg_names = "";
        std::string arg_types = "";
        for (int32_t i = 0; i < arg_num; ++i) {
          //T_PARAM_DEFINITION
          ParseNode *element = function_element_list_node->children_[i];
          //T_IDENT
          ParseNode *arg_name_node = element->children_[0];
          //get arg_name
          const char* arg_name = arg_name_node->str_value_;
          arg_names += arg_name;
          if (i != arg_num - 1) arg_names += ",";
          //get arg type
          ParseNode *type_node = element->children_[1];
          switch (type_node->value_) {
              case 1:
                arg_types += "STRING";
                break;
              case 2:
                arg_types += "INTEGER";
                break;
              case 3:
                arg_types += "REAL";
                break;
              case 4:
                arg_types += "DECIMAL";
                break;       
          }
          if (i != arg_num - 1) arg_types += ",";
        }
        //set arg_names
        create_udf_model_arg.udf_model_.set_arg_names(common::ObString(arg_names.length(),arg_names.c_str()));
        //set arg_types
        create_udf_model_arg.udf_model_.set_arg_types(common::ObString(arg_types.length(),arg_types.c_str()));
        //set return type
        switch (create_udf_model_node->children_[3]->value_) {
          case 1:
            create_udf_model_arg.udf_model_.set_ret(schema::ObPythonUDF::STRING);
            break;
          case 2:
            create_udf_model_arg.udf_model_.set_ret(schema::ObPythonUDF::INTEGER);
            break;
          case 3:
            create_udf_model_arg.udf_model_.set_ret(schema::ObPythonUDF::REAL);
            break;
          case 4:
            create_udf_model_arg.udf_model_.set_ret(schema::ObPythonUDF::DECIMAL);
            break;
        }
        //resolve model metadata list
        ParseNode *model_metadata_list_node = create_udf_model_node->children_[4];
        if (OB_ISNULL(model_metadata_list_node)
            || T_METADATA_LIST != model_metadata_list_node->type_
            || 3 < model_metadata_list_node->num_child_) {
          ret = OB_INVALID_ARGUMENT;
          SQL_RESV_LOG(WARN, "invalid argument.", K(ret));
        } else {    //解析模型元数据信息
            //get model_metadata_node
            int model_metadata_num = model_metadata_list_node->num_child_;
            std::string framework, model_type;
            ObString model_path;
            if (model_metadata_num > 0) {
              for (int i = 0; i < model_metadata_num; i++) {
                ParseNode *model_metadata_node = model_metadata_list_node->children_[i];
                switch (model_metadata_node->value_) {
                  case 1:{
                    //get framework
                    ParseNode *framework_node = model_metadata_node->children_[0];
                    framework = framework_node->str_value_;
                    break;
                  }
                  case 2:{
                    //get model type
                    ParseNode *model_type_node = model_metadata_node->children_[0];
                    model_type = model_type_node->str_value_;
                    break;
                  }
                  case 3:{
                    //get model path
                    ParseNode *model_path_node = model_metadata_node->children_[0];
                    model_path = ObString(model_path_node->str_len_, model_path_node->str_value_);
                    break;       
                  }
                }
              }
            }
            //转为大写
            std::transform(framework.begin(), framework.end(), framework.begin(), ::toupper);
            std::transform(model_type.begin(), model_type.end(), model_type.begin(), ::toupper);
            //set framework
            if (framework == "ONNX"){
                create_udf_model_arg.udf_model_.set_framework(schema::ObUdfModel::ONNX);
            } else if (framework == "SKLEARN") {
                create_udf_model_arg.udf_model_.set_framework(schema::ObUdfModel::SKLEARN);
            } else if (framework == "PYTORCH") {
                create_udf_model_arg.udf_model_.set_framework(schema::ObUdfModel::PYTORCH);
            } else {
                create_udf_model_arg.udf_model_.set_framework(schema::ObUdfModel::UNSUPPORTED);
            }
            //set model_type          
            if (model_type == "DESISION_TREE"){
                create_udf_model_arg.udf_model_.set_model_type(schema::ObUdfModel::DECISION_TREE);
            } else {
                create_udf_model_arg.udf_model_.set_model_type(schema::ObUdfModel::UNSUPPORTED);
            }
            //set model_path
            create_udf_model_arg.udf_model_.set_model_path(model_path);     
        }

        //set tenant_id
        create_udf_model_arg.udf_model_.set_tenant_id(params_.session_info_->get_effective_tenant_id());
      }
    }            
    return ret;
}
}
}
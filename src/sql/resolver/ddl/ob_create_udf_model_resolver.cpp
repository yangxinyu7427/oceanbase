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
        || 3 != create_udf_model_node->num_child_         //语法树根节点的孩子数不正确
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
        //resolve model metadata list
        ParseNode *model_metadata_list_node = create_udf_model_node->children_[2];
        if (OB_ISNULL(model_metadata_list_node)
            || T_METADATA_LIST != model_metadata_list_node->type_
            || 1 != model_metadata_list_node->num_child_         //模型元数据节点的孩子数不正确
            || OB_ISNULL(model_metadata_list_node->children_)) {
          ret = OB_INVALID_ARGUMENT;
          SQL_RESV_LOG(WARN, "invalid argument.", K(ret));
        } else {    //解析模型元数据信息
            //get model_metadata_node
            ParseNode *model_metadata_node = model_metadata_list_node->children_[0];
            //get framework
            ParseNode *framework_node = model_metadata_node->children_[0];
            const char* framework = framework_node->str_value_;
            //修改比较
            //set framework
            if (std::strcmp(framework, "ONNX") == 0){
                create_udf_model_arg.udf_model_.set_framework(schema::ObUdfModel::ONNX);
            } else if (std::strcmp(framework, "SKLEARN") == 0) {
                create_udf_model_arg.udf_model_.set_framework(schema::ObUdfModel::SKLEARN);
            } else if (std::strcmp(framework, "PYTORCH") == 0) {
                create_udf_model_arg.udf_model_.set_framework(schema::ObUdfModel::PYTORCH);
            } else {
                create_udf_model_arg.udf_model_.set_framework(schema::ObUdfModel::UNSUPPORTED);
            }
            //get model type
            ParseNode *model_type_node = model_metadata_node->children_[1];
            //set model_type          
            const char* model_type = model_type_node->str_value_;
            if (std::strcmp(model_type, "decision_tree") == 0){
                create_udf_model_arg.udf_model_.set_framework(schema::ObUdfModel::DECISION_TREE);
            } else {
                create_udf_model_arg.udf_model_.set_framework(schema::ObUdfModel::UNSUPPORTED);
            }
            ObString model_path;
            //get model path
            ParseNode *model_path_node = model_metadata_node->children_[2];
            model_path = ObString(model_path_node->str_len_, model_path_node->str_value_);
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
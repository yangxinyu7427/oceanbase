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
#include "sql/resolver/ddl/ob_alter_udf_model_resolver.h"
namespace oceanbase
{
namespace sql
{
ObAlterUdfModelResolver::ObAlterUdfModelResolver(ObResolverParams &params)
    : ObDDLResolver(params)
{
}
ObAlterUdfModelResolver::~ObAlterUdfModelResolver()
{
}
int ObAlterUdfModelResolver::resolve(const ParseNode &parse_tree)
{
    int ret = OB_SUCCESS;
    ParseNode *alter_udf_model_node = const_cast<ParseNode*>(&parse_tree);
    if (OB_ISNULL(alter_udf_model_node)
        || T_ALTER_UDF_MODEL != alter_udf_model_node->type_
        || 1 > alter_udf_model_node->num_child_         //语法树根节点的孩子数不正确
        || OB_ISNULL(alter_udf_model_node->children_)) {
      ret = OB_INVALID_ARGUMENT;
      SQL_RESV_LOG(WARN, "invalid argument.", K(ret));
    } else {
      ObAlterUdfModelStmt *alter_udf_model_stmt = NULL;
      ObString model_name, model_path_before, model_path_after;
      if (OB_ISNULL(alter_udf_model_stmt = create_stmt<ObAlterUdfModelStmt>())) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        SQL_RESV_LOG(ERROR, "failed to create alter_udf_model_stmt", K(ret));
      } else {
        stmt_ = alter_udf_model_stmt;
        obrpc::ObAlterUdfModelArg &alter_udf_model_arg = alter_udf_model_stmt->get_alter_udf_model_arg();
        //get model name
        ParseNode *relation_node = alter_udf_model_node->children_[0];
        model_name = ObString(relation_node->str_len_, relation_node->str_value_);
        //set model name
        alter_udf_model_arg.model_name_ = model_name;
        
        //resolve model compression method
        ParseNode *compression_method_node = alter_udf_model_node->children_[1];
        if (T_DISTILLATION == compression_method_node->type_) {
          alter_udf_model_arg.is_distillation_ = true;
        } else if (T_BINARIZATION == compression_method_node->type_) {
          alter_udf_model_arg.is_binarization_ = true;
        }
        //resolve model path
        ParseNode *model_path_before_node = alter_udf_model_node->children_[2];
        model_path_before = ObString(model_path_before_node->str_len_, model_path_before_node->str_value_);
        alter_udf_model_arg.model_path_before_ = model_path_before;

        ParseNode *model_path_after_node = alter_udf_model_node->children_[3];
        model_path_after = ObString(model_path_after_node->str_len_, model_path_after_node->str_value_);
        alter_udf_model_arg.model_path_after_ = model_path_after;

        //set tenant_id
        alter_udf_model_arg.tenant_id_ = params_.session_info_->get_effective_tenant_id();
      }
    }
    return ret;
}
}
}
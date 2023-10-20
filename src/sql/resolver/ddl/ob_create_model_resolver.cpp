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
#include "sql/resolver/ddl/ob_create_model_resolver.h"

namespace oceanbase
{
namespace sql
{

ObCreateModelResolver::ObCreateModelResolver(ObResolverParams &params)
    : ObDDLResolver(params)
{
}

ObCreateModelResolver::~ObCreateModelResolver()
{
}

int ObCreateModelResolver::resolve(const ParseNode &parse_tree)
{
    int ret = OB_SUCCESS;
    ParseNode *create_model_node = const_cast<ParseNode*>(&parse_tree);
    if (OB_ISNULL(create_model_node)
        || T_CREATE_MODEL != create_model_node->type_
        || 4 != create_model_node->num_child_         //语法树根节点的孩子数不正确
        || OB_ISNULL(create_model_node->children_)) {
      ret = OB_INVALID_ARGUMENT;
      SQL_RESV_LOG(WARN, "invalid argument.", K(ret));
    } else {
      ObCreateModelStmt *create_model_stmt = NULL;
      ObString model_name;
      if (OB_ISNULL(create_model_stmt = create_stmt<ObCreateModelStmt>())) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        SQL_RESV_LOG(ERROR, "failed to create create_model_stmt", K(ret));
      } else {
        stmt_ = create_model_stmt;
        obrpc::ObCreateModelArg &create_model_arg = create_model_stmt->get_create_model_arg();
        
        //get model name
        ParseNode *relation_node = create_model_node->children_[0];
        model_name = ObString(relation_node->str_len_, relation_node->str_value_);
        //set model name
        create_model_arg.python_udf_.set_name(model_name);
        //resolve function element
        ParseNode *function_element_list_node = create_model_node->children_[1];
        //set arg_num
        int arg_num = function_element_list_node->num_child_;
        create_model_arg.python_udf_.set_arg_num(arg_num);
        for (int32_t i = 0; i < arg_num; ++i) {
            //T_PARAM_DEFINITION
            ParseNode *element = function_element_list_node->children_[i];
            //T_IDENT
            ParseNode *arg_name_node = element->children_[0];
            //get arg_name
            const char* arg_name = arg_name_node->str_value_;
            create_model_arg.python_udf_.set_arg_names(arg_name);
            //arg type
            ParseNode *type_node = element->children_[1];
            switch (type_node->value_) {
                case 1:
                    create_model_arg.python_udf_.set_arg_types("STRING");
                    break;
                case 2:
                    create_model_arg.python_udf_.set_arg_types("INTEGER");
                    break;
                case 3:
                    create_model_arg.python_udf_.set_arg_types("REAL");
                    break;
                case 4:
                    create_model_arg.python_udf_.set_arg_types("DECIMAL");
                    break;       
            }
        }
        //set return type
        switch (create_model_node->children_[2]->value_) {
            case 1:
                create_model_arg.python_udf_.set_ret(schema::ObPythonUDF::STRING);
                break;
            case 2:
                create_model_arg.python_udf_.set_ret(schema::ObPythonUDF::INTEGER);
                break;
            case 3:
                create_model_arg.python_udf_.set_ret(schema::ObPythonUDF::REAL);
                break;
            case 4:
                create_model_arg.python_udf_.set_ret(schema::ObPythonUDF::DECIMAL);
                break;
        }
        //set pycall
        create_model_arg.python_udf_.set_pycall(ObString(create_model_node->children_[3]->str_len_, create_model_node->children_[3]->str_value_));
        //set tenant_id
        create_model_arg.python_udf_.set_tenant_id(params_.session_info_->get_effective_tenant_id());
      }
    }            
    return ret;
}

}
}
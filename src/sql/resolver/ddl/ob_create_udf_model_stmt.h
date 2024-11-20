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
#ifndef _OB_CREATE_UDF_MODEL_STMT_H
#define _OB_CREATE_UDF_MODEL_STMT_H 1
#include "sql/resolver/ddl/ob_ddl_stmt.h"
namespace oceanbase
{
namespace sql
{
class ObCreateUdfModelStmt : public ObDDLStmt
{
public:
    ObCreateUdfModelStmt() :
        ObDDLStmt(stmt::T_CREATE_UDF_MODEL)
    {}
    ~ObCreateUdfModelStmt() {}
    obrpc::ObCreateUdfModelArg &get_create_udf_model_arg() {return create_udf_model_arg_; }
    virtual obrpc::ObDDLArg &get_ddl_arg() {return create_udf_model_arg_; };
private:
    obrpc::ObCreateUdfModelArg create_udf_model_arg_;  
    DISALLOW_COPY_AND_ASSIGN(ObCreateUdfModelStmt);  
};
}
}
#endif /* _OB_CREATE_UDF_MODEL_STMT_H */
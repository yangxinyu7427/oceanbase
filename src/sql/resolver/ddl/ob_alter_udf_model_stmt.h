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
#ifndef _OB_ALTER_UDF_MODEL_STMT_H
#define _OB_ALTER_UDF_MODEL_STMT_H 1
#include "sql/resolver/ddl/ob_ddl_stmt.h"
namespace oceanbase
{
namespace sql
{
class ObAlterUdfModelStmt : public ObDDLStmt
{
public:
  ObAlterUdfModelStmt() :
      ObDDLStmt(stmt::T_ALTER_UDF_MODEL)
  {}
  ~ObAlterUdfModelStmt() { }
  obrpc::ObAlterUdfModelArg &get_alter_udf_model_arg() { return alter_udf_model_arg_; }
  obrpc::ObDDLArg &get_ddl_arg() { return alter_udf_model_arg_; }
  TO_STRING_KV(K_(alter_udf_model_arg));
private:
  obrpc::ObAlterUdfModelArg alter_udf_model_arg_;
  DISALLOW_COPY_AND_ASSIGN(ObAlterUdfModelStmt);
};
}
}
#endif /* _OB_ALTER_UDF_MODEL_STMT_H */
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
#ifndef _OB_DROP_MODEL_STMT_H
#define _OB_DROP_MODEL_STMT_H 1
#include "sql/resolver/ddl/ob_ddl_stmt.h"
namespace oceanbase
{
namespace sql
{
class ObDropUdfModelStmt : public ObDDLStmt
{
public:
  ObDropUdfModelStmt() :
      ObDDLStmt(stmt::T_DROP_UDF_MODEL),
      drop_udf_model_arg_()
  {}
  ~ObDropUdfModelStmt() { }
  void set_model_name(const common::ObString &model_name) { drop_udf_model_arg_.name_ = model_name; }
  void set_tenant_id(uint64_t tenant_id) { drop_udf_model_arg_.tenant_id_ = tenant_id; }
  obrpc::ObDropUdfModelArg &get_drop_udf_model_arg() { return drop_udf_model_arg_; }
  obrpc::ObDDLArg &get_ddl_arg() { return drop_udf_model_arg_; }
  TO_STRING_KV(K_(drop_udf_model_arg));
private:
  obrpc::ObDropUdfModelArg drop_udf_model_arg_;
  DISALLOW_COPY_AND_ASSIGN(ObDropUdfModelStmt);
};
}
}
#endif /* _OB_DROP_UDF_MODEL_STMT_H */
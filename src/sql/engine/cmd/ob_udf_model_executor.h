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
#ifndef OB_MODEL_EXECUTOR_H_
#define OB_MODEL_EXECUTOR_H_
#include "common/object/ob_object.h"
#include "lib/container/ob_se_array.h"
namespace oceanbase
{
namespace common
{
class ObIAllocator;
class ObExprCtx;
// namespace sqlclient
// {
class ObMySQLProxy;
// }
}
namespace sql
{
class ObExecContext;
class ObRawExpr;
class ObCreateUdfModelStmt;
class ObCreateUdfModelExecutor
{
public:
  const static int OB_DEFAULT_ARRAY_SIZE = 16;
  ObCreateUdfModelExecutor(){}
  virtual ~ObCreateUdfModelExecutor(){}
  int execute(ObExecContext &ctx, ObCreateUdfModelStmt &stmt);
private:
  DISALLOW_COPY_AND_ASSIGN(ObCreateUdfModelExecutor);
};
class ObDropUdfModelStmt;
class ObDropUdfModelExecutor
{
public:
  ObDropUdfModelExecutor(){}
  virtual ~ObDropUdfModelExecutor(){}
  int execute(ObExecContext &ctx, ObDropUdfModelStmt &stmt);
private:
  DISALLOW_COPY_AND_ASSIGN(ObDropUdfModelExecutor);
};
}
}
#endif
/**
 * author: xyyang
 * find redundancy of python udf and then remove them
 */
#ifndef OB_TRANSFORM_PYUDF_REDUNDENT_H
#define OB_TRANSFORM_PYUDF_REDUNDENT_H

#include "sql/rewrite/ob_transform_rule.h"
#include "sql/resolver/dml/ob_select_stmt.h"
#include "sql/ob_select_stmt_printer.h"

namespace oceanbase
{
namespace sql
{

class ObTransformPyUDFRedundent : public ObTransformRule
{
public:
  ObTransformPyUDFRedundent(ObTransformerCtx *ctx);

  virtual ~ObTransformPyUDFRedundent();

  virtual int transform_one_stmt(
    ObIArray<ObParentDMLStmt> &parent_stmts,
    ObDMLStmt *&stmt,
    bool &trans_happened) override;

private:
  virtual int need_transform(
    const common::ObIArray<ObParentDMLStmt> &parent_stmts,
    const int64_t current_level,
    const ObDMLStmt &stmt,
    bool &need_trans) override;

  virtual int construct_transform_hint(ObDMLStmt &stmt, void *trans_params) override;


private:
  ObArenaAllocator allocator_;

};

}
}

#endif // OB_TRANSFORM_PYUDF_REDUNDENT_H
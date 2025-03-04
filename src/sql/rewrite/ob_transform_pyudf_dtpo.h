#ifndef OCEANBASE_SQL_REWRITE_OB_TRANSFORM_PYUDF_DTPO_
#define OCEANBASE_SQL_REWRITE_OB_TRANSFORM_PYUDF_DTPO_
#include "sql/rewrite/ob_transform_rule.h"
namespace oceanbase
{
namespace sql
{
class ObTransformPyUdfDTPO : public ObTransformRule {
public:
  explicit ObTransformPyUdfDTPO(ObTransformerCtx *ctx)
    : ObTransformRule(ctx, TransMethod::PRE_ORDER, T_PYUDF_DTPO) {}
  virtual ~ObTransformPyUdfDTPO() {}
  virtual int transform_one_stmt(common::ObIArray<ObParentDMLStmt> &parent_stmts,
                                 ObDMLStmt *&stmt,
                                 bool &trans_happened) override;
  virtual int transform_one_stmt_with_outline(common::ObIArray<ObParentDMLStmt> &parent_stmts,
                                              ObDMLStmt *&stmt,
                                              bool &trans_happened) override;
private:
  int onnx_decision_tree_prune(ObDMLStmt *stmt);

  int do_recursive_onnx_decision_tree_prune(ObSelectStmt *select_stmt,
                                            ObRawExpr *parent_expr,
                                            ObRawExpr *cmp_expr);

  virtual int need_transform(
    const common::ObIArray<ObParentDMLStmt> &parent_stmts,
    const int64_t current_level,
    const ObDMLStmt &stmt,
    bool &need_trans) override;

  virtual int construct_transform_hint(ObDMLStmt &stmt, void *trans_params) override;
private:
  DISALLOW_COPY_AND_ASSIGN(ObTransformPyUdfDTPO);
};
}
}
#endif
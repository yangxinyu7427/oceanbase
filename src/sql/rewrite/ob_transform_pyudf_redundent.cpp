/**
 * author: xyyang
 * find redundancy of python udf and then change them
 */
#define USING_LOG_PREFIX SQL_REWRITE
#include <regex>
#include <string>
#include <map>
#include "sql/rewrite/ob_transform_pyudf_redundent.h"
#include "sql/rewrite/ob_stmt_comparer.h"
#include "sql/rewrite/ob_transform_utils.h"
#include "sql/optimizer/ob_optimizer_util.h"
#include "sql/resolver/expr/ob_raw_expr_util.h"
#include "sql/rewrite/ob_predicate_deduce.h"
#include "share/schema/ob_table_schema.h"
#include "common/ob_smart_call.h"
#include "sql/rewrite/onnx_optimizer/onnxoptimizer/optimize_c_api/optimize_c_api.h"
// #include "sql/engine/expr/ob_expr_python_udf.h"

#include "objit/include/objit/expr/ob_iraw_expr.h"
#include "sql/resolver/expr/ob_raw_expr.h"

#include "sql/ob_select_stmt_printer.h"
#include "deps/oblib/src/lib/json/ob_json_print_utils.h"

using namespace oceanbase::sql;
using namespace oceanbase::common;

string opted_model_path="/root/onnx_output/model_opted.onnx";


ObTransformPyUDFRedundent::ObTransformPyUDFRedundent(ObTransformerCtx *ctx)
    : ObTransformRule(ctx, TransMethod::POST_ORDER, T_PYUDF_REDUNDENT),
      allocator_("PyUDFRedundent")
{}

ObTransformPyUDFRedundent::~ObTransformPyUDFRedundent()
{}

int ObTransformPyUDFRedundent::transform_one_stmt(
  common::ObIArray<ObParentDMLStmt> &parent_stmts, ObDMLStmt *&stmt, bool &trans_happened)
{
  int ret = OB_SUCCESS;
  trans_happened = false;
  ObSQLSessionInfo* session=ctx_->exec_ctx_->get_my_session();
  LOG_TRACE("Run transform ObTransformPyUDFRedundent", K(ret));
  ObSelectStmt *select_stmt = NULL;
  string onnx_model_opted_path;
  if (OB_ISNULL(stmt) || OB_ISNULL(ctx_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("stmt is null", K(ret), K(stmt), K(ctx_));
  } else if (!stmt->is_select_stmt()) {
    // do nothing
    OPT_TRACE("not select stmt, can not transform");
  } else if (FALSE_IT(select_stmt = static_cast<ObSelectStmt*>(stmt))) {
    //准备进行改写
    LOG_WARN("select stmt is NULL", K(ret));
  } else if (select_stmt->get_condition_exprs().empty()) {
    //没有改写空间
    LOG_WARN("input preds is empty", K(ret));
  } 
  else if (OB_FAIL()){

  }
  else{
    trans_happened = true;
    stmt = select_stmt;
  }
  return ret;  

}



int ObTransformPyUDFRedundent::need_transform(const common::ObIArray<ObParentDMLStmt> &parent_stmts,
  const int64_t current_level,
  const ObDMLStmt &stmt,
  bool &need_trans)
{
  int ret = OB_SUCCESS;
  LOG_TRACE("Check need transform of ObTransformPyUDFRedundent.", K(ret));
  need_trans = false;
  ObSEArray<ObSelectStmt*, 16> child_stmts;
  int python_udf_count=0;

  for(int32_t i = 0; i < stmt.get_condition_size(); i++) {
    python_udf_count=python_udf_count+ObTransformUtils::count_python_udf_num(const_cast<ObRawExpr *>(stmt.get_condition_expr(i)));
  }

  if(python_udf_count>=1){
    need_trans = true;
    LOG_DEBUG("this query need transform of ObTransformPyUDFRedundent ,udf count is ",K(python_udf_count));
  }
  return ret;
}

int ObTransformPyUDFRedundent::construct_transform_hint(ObDMLStmt &stmt, void *trans_params) 
{
  int ret = OB_SUCCESS;
  ObIArray<ObSelectStmt*> *transed_stmts = static_cast<ObIArray<ObSelectStmt*>*>(trans_params);
  if (OB_ISNULL(ctx_) || OB_ISNULL(ctx_->allocator_) || OB_ISNULL(transed_stmts)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected null", K(ret), K(ctx_));
  } else if (OB_FAIL(ctx_->add_src_hash_val(ctx_->src_qb_name_))) {
    LOG_WARN("failed to add src hash val", K(ret));
  } else {
    ObTransHint *hint = NULL;
    ObString qb_name;
    ObSelectStmt *cur_stmt = NULL;
    for (int64_t i = 0; OB_SUCC(ret) && i < transed_stmts->count(); i++) {
      if (OB_ISNULL(cur_stmt = transed_stmts->at(i))) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("transed_stmt is null", K(ret), K(i));
      } else if (OB_FAIL(ctx_->add_used_trans_hint(get_hint(cur_stmt->get_stmt_hint())))) {
        LOG_WARN("failed to add used hint", K(ret));
      } else if (OB_FAIL(ObQueryHint::create_hint(ctx_->allocator_, get_hint_type(), hint))) {
        LOG_WARN("failed to create hint", K(ret));
      } else if (OB_FAIL(ctx_->outline_trans_hints_.push_back(hint))) {
        LOG_WARN("failed to push back hint", K(ret));
      } else if (OB_FAIL(cur_stmt->get_qb_name(qb_name))) {
        LOG_WARN("failed to get qb name", K(ret));
      } else if (OB_FAIL(cur_stmt->adjust_qb_name(ctx_->allocator_,
                                                  qb_name,
                                                  ctx_->src_hash_val_))) {
        LOG_WARN("adjust stmt id failed", K(ret));
      } else {
        hint->set_qb_name(qb_name);
      }
    }
  }
  return ret;
}
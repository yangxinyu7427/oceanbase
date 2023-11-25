/**
 * author: xyyang
 * find redundancy of python udf and then remove them
 */
#define USING_LOG_PREFIX SQL_REWRITE
#include "sql/rewrite/ob_transform_pyudf_merge.h"
#include "sql/rewrite/ob_stmt_comparer.h"
#include "sql/rewrite/ob_transform_utils.h"
#include "sql/optimizer/ob_optimizer_util.h"
#include "sql/resolver/expr/ob_raw_expr_util.h"
#include "sql/rewrite/ob_predicate_deduce.h"
#include "share/schema/ob_table_schema.h"
#include "common/ob_smart_call.h"

#include "objit/include/objit/expr/ob_iraw_expr.h"
#include "sql/resolver/expr/ob_raw_expr.h"

#include "sql/ob_select_stmt_printer.h"
#include "deps/oblib/src/lib/json/ob_json_print_utils.h"

using namespace oceanbase::sql;
using namespace oceanbase::common;

ObTransformPyUDFMerge::ObTransformPyUDFMerge(ObTransformerCtx *ctx)
    : ObTransformRule(ctx, TransMethod::POST_ORDER, T_PYUDF_MERGE),
      allocator_("PyUDFMerge")
{}

ObTransformPyUDFMerge::~ObTransformPyUDFMerge()
{}

int ObTransformPyUDFMerge::transform_one_stmt(
  common::ObIArray<ObParentDMLStmt> &parent_stmts, ObDMLStmt *&stmt, bool &trans_happened)
{
  int ret = OB_SUCCESS;
  trans_happened = false;
  ObSelectStmt *select_stmt = NULL;
  int pyudf_count=0;
  ObSEArray<int32_t, 20> pyudf_expr_index;

  LOG_TRACE("Run transform ObTransformPyUDFMerge", K(ret));
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
  } else if(OB_FAIL(ObTransformUtils::extract_python_udf_exprs_idx_in_condition(pyudf_expr_index, select_stmt->get_condition_exprs()))){
    LOG_WARN("extract python_udf_exprs index fail", K(ret));
  } else if(OB_FAIL(merge_python_udf_expr_in_condition(pyudf_expr_index, select_stmt->get_condition_exprs()))){
    LOG_WARN("merge python_udf_exprs fail", K(ret));
  } else{
    trans_happened = true;
    stmt = select_stmt;
  }
  return ret;  

}

int ObTransformPyUDFMerge::merge_python_udf_expr_in_condition(
  ObIArray<int32_t> &idx_list,
  ObIArray<ObRawExpr *> &src_exprs)
{
  int ret = OB_SUCCESS;
  // test code
  for (int i=1;i<idx_list.count();i++){
    src_exprs.remove(i);
  }
  return ret;
}


int ObTransformPyUDFMerge::need_transform(const common::ObIArray<ObParentDMLStmt> &parent_stmts,
  const int64_t current_level,
  const ObDMLStmt &stmt,
  bool &need_trans)
{
  int ret = OB_SUCCESS;
  LOG_TRACE("Check need transform of ObTransformPyUDFMerge.", K(ret));
  need_trans = false;
  int pyudf_count=0;

  for(int32_t i = 0; i < stmt.get_condition_size(); i++) 
    if(ObTransformUtils::expr_contain_type(const_cast<ObRawExpr *>(stmt.get_condition_expr(i)), T_FUN_SYS_PYTHON_UDF)) 
      pyudf_count++;

  if(pyudf_count>1) need_trans=true;

  return ret;
}

int ObTransformPyUDFMerge::construct_transform_hint(ObDMLStmt &stmt, void *trans_params) 
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
/**
 * author: xyyang
 * find redundancy of python udf and then remove them
 */
#define USING_LOG_PREFIX SQL_REWRITE
#include <regex>
#include <string>
#include "sql/rewrite/ob_transform_pyudf_merge.h"
#include "sql/rewrite/ob_stmt_comparer.h"
#include "sql/rewrite/ob_transform_utils.h"
#include "sql/optimizer/ob_optimizer_util.h"
#include "sql/resolver/expr/ob_raw_expr_util.h"
#include "sql/rewrite/ob_predicate_deduce.h"
#include "share/schema/ob_table_schema.h"
#include "common/ob_smart_call.h"
#include "sql/rewrite/onnx_optimizer_C/onnxoptimizer/optimize_c_api/optimize_c_api.h"
// #include "sql/engine/expr/ob_expr_python_udf.h"

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
  ObSEArray<int64_t, 4> pyudf_expr_index;
  ObSEArray<ObPythonUdfRawExpr *,4> python_udf_expr_list;
  ObSEArray<oceanbase::share::schema::ObPythonUDFMeta,4> python_udf_meta_list;
  // SQL_LOG(DEBUG, "this is stmt before ObTransformPyUDFMerge", "query", SJ(*stmt));
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
  } else if(OB_FAIL(extract_python_udf_expr_in_condition(python_udf_expr_list,select_stmt->get_condition_exprs(),python_udf_meta_list))){
    LOG_WARN("extract python_udf_expr in condition fail", K(ret));
  } else if(OB_FAIL(ObTransformUtils::extract_python_udf_exprs_idx_in_condition(pyudf_expr_index, select_stmt->get_condition_exprs()))){
    LOG_WARN("extract python_udf_exprs index fail", K(ret));
  } else if(OB_FAIL(merge_python_udf_expr_in_condition(pyudf_expr_index, select_stmt->get_condition_exprs()))){
    LOG_WARN("merge python_udf_exprs fail", K(ret));
  } else{
    trans_happened = true;
    stmt = select_stmt;
  }
  // test code
  ObSEArray<ObString, 4> onnx_model_path_list;
  string out_path="/root/onnx_output/model_opted.onnx";
  // ObString onnx_model_path;
  // if(OB_FAIL(get_onnx_model_path_from_python_udf_meta(onnx_model_path,python_udf_meta_list.at(0)))){
  //   LOG_WARN("get_onnx_model_path_from_python_udf_meta fail", K(ret));
  // }
  if(OB_FAIL(get_onnx_model_path_list_from_python_udf_meta_list(onnx_model_path_list, python_udf_meta_list))){
    LOG_WARN("get_onnx_model_path_list_from_python_udf_meta_list fail", K(ret));
  } else if(OB_FAIL(merge_onnx_model_with_path_list(onnx_model_path_list, out_path))){
    LOG_WARN("merge_onnx_model_with_path_list fail", K(ret));
  } else{
    LOG_DEBUG("merge onnx model success, the output model path is ",K(ObString(out_path.c_str())));
  }

  // string path1="/root/onnx_input/model_lr.onnx";
  // string path2="/root/onnx_input/model_linear.onnx";
  // string pre1="model_lr";
  // string pre2="model_linear";
  // string out_path="/root/onnx_output/model_opted.onnx";
  // optimize_with_model_path(path1,path2,pre1,pre2,out_path);
  // LOG_DEBUG("this is python udf count",K(python_udf_meta_list.count()));
  // for(int i=0;i<python_udf_meta_list.count();i++){
  //   LOG_DEBUG("this is python udf",K(python_udf_meta_list.at(i).pycall_));
  // }
  return ret;  

}

int ObTransformPyUDFMerge::merge_onnx_model_with_path_list(
  ObIArray<ObString> &onnx_model_path_list,
  string out_path){
    int ret = OB_SUCCESS;
    string path1(onnx_model_path_list.at(0).ptr());
    string path2(onnx_model_path_list.at(1).ptr());
    string pre1="1";
    string pre2="2";
    try{
      optimize_with_model_path(path1,path2,pre1,pre2,out_path);
    } catch(...){
      LOG_WARN("optimize_with_model_path fail");
      ret=OB_ERROR;
    }
    return ret;
}

int ObTransformPyUDFMerge::get_onnx_model_path_list_from_python_udf_meta_list(
  ObIArray<ObString> &onnx_model_path_list,
  ObIArray<oceanbase::share::schema::ObPythonUDFMeta > &python_udf_meta_list){
  int ret = OB_SUCCESS;
  for(int i=0;i<python_udf_meta_list.count();i++){
    ObString onnx_model_path;
    if(OB_FAIL(get_onnx_model_path_from_python_udf_meta(onnx_model_path, python_udf_meta_list.at(i)))){
      LOG_WARN("get_onnx_model_path_from_python_udf_meta fail", K(ret));
    } else{
      onnx_model_path_list.push_back(onnx_model_path);
    }
  }
  return ret;
}

int ObTransformPyUDFMerge::get_onnx_model_path_from_python_udf_meta(ObString &onnx_model_path, oceanbase::share::schema::ObPythonUDFMeta &python_udf_meta){
  int ret =OB_SUCCESS;
  // there is bug
  // //pycall
  // std::string pycall(python_udf_meta.pycall_.ptr());
  // std::string pGetModelPath="\ndef pygetmodelpath():\
  // \n\treturn onnx_path\0";
  // pycall.append(pGetModelPath);

  // //runtime variables
  // const char* bytes=NULL;
  // PyObject *pModule = NULL;
  // PyObject *dic = NULL;
  // PyObject *v = NULL;
  // PyObject *pInitial = NULL;
  // PyObject *pGetModel = NULL;
  // PyObject *pResult = NULL;
  // PyObject* unicode = NULL;
  // const char* pycall_c = pycall.c_str();

  // // // 初始化 Python 解释器
  // // Py_Initialize();
  // //Acquire GIL
  // bool nStatus = PyGILState_Check();
  // PyGILState_STATE gstate;
  // if(!nStatus) {
  //   gstate = PyGILState_Ensure();
  //   nStatus = true;
  // }

  // // prepare and import python code
  // pModule = PyImport_AddModule("__main__"); // load main module
  // if(OB_ISNULL(pModule)) {
  //   ret = OB_ERR_UNEXPECTED;
  //   LOG_WARN("fail to import main module", K(ret));
  //   goto destruction;
  // }
  // dic = PyModule_GetDict(pModule); // get main module dic
  // if(OB_ISNULL(dic)) {
  //   ret = OB_ERR_UNEXPECTED;
  //   LOG_WARN("fail to get main module dic", K(ret));
  //   goto destruction;
  // } 
  // v = PyRun_StringFlags(pycall_c, Py_file_input, dic, dic, NULL); // test pycall
  // if(OB_ISNULL(v)) {
  //   ObExprPythonUdf::process_python_exception();
  //   ret = OB_ERR_UNEXPECTED;
  //   LOG_WARN("fail to write pycall into module", K(ret));
  //   goto destruction;
  // }
  // pInitial = PyObject_GetAttrString(pModule, "pyinitial"); // get pyInitial()
  // if(OB_ISNULL(pInitial) || !PyCallable_Check(pInitial)) {
  //   ObExprPythonUdf::process_python_exception();
  //   ret = OB_ERR_UNEXPECTED;
  //   LOG_WARN("Fail to import pyinitial", K(ret));
  //   goto destruction;
  // } else if (OB_ISNULL(PyObject_CallObject(pInitial, NULL))){
  //   ObExprPythonUdf::process_python_exception();
  //   ret = OB_ERR_UNEXPECTED;
  //   LOG_WARN("Fail to run pyinitial", K(ret));
  //   goto destruction;
  // } else {
  //   LOG_DEBUG("Import python udf pyinitial", K(ret));
  // }

  // pGetModel = PyObject_GetAttrString(pModule, "pygetmodelpath"); // get pygetmodelpath()
  // if(OB_ISNULL(pGetModel) || !PyCallable_Check(pGetModel)) {
  //   ObExprPythonUdf::process_python_exception();
  //   ret = OB_ERR_UNEXPECTED;
  //   LOG_WARN("Fail to import pGetModel", K(ret));
  //   goto destruction;
  // } else if (OB_ISNULL(PyObject_CallObject(pGetModel, NULL))){
  //   ObExprPythonUdf::process_python_exception();
  //   ret = OB_ERR_UNEXPECTED;
  //   LOG_WARN("Fail to run pGetModel", K(ret));
  //   goto destruction;
  // } else {
  //   LOG_DEBUG("Import python udf pGetModel", K(ret));
  // }
  // pResult=PyObject_CallObject(pGetModel, NULL);
  // if (pResult != NULL) {
  //       if (PyUnicode_Check(pResult)) { // 检查返回值是否是 Python 字符串
  //           unicode = PyUnicode_AsUTF8String(pResult); // 将 Python 字符串对象转换为 UTF-8 编码的字节对象
  //           if (unicode != NULL) {
  //               bytes = PyBytes_AsString(unicode); // 将字节对象转换为 C 字符串
  //               if (bytes != NULL) {
  //                   onnx_model_path = ObString(bytes); // 将 C 字符串转换为 C++ 字符串
  //               }
  //           }
  //       } else {
  //           LOG_WARN("Fail to get unicode", K(ret)); // 处理非字符串对象
  //       }
  //   } else {
  //       LOG_WARN("Fail to get pResult", K(ret)); // 处理函数调用失败
  //   }
  
  // destruction: 
  // // 释放 Python 对象
  // Py_XDECREF(dic);
  // Py_XDECREF(pModule);
  // Py_XDECREF(v);
  // Py_XDECREF(pInitial);
  // Py_XDECREF(pGetModel);
  // Py_XDECREF(pResult);
  // Py_XDECREF(unicode);
  // //release GIL
  // if(nStatus)
  //   PyGILState_Release(gstate);
  // LOG_DEBUG("get model path", K(onnx_model_path));
  // // // 关闭 Python 解释器
  // // Py_FinalizeEx();
  string pycall(python_udf_meta.pycall_.ptr());
  std::regex pattern("onnx_path='(.*?)'");
  std::smatch match;
  if (std::regex_search(pycall, match, pattern)) {
        if (match.size() > 1) {
          onnx_model_path=ObString(match[1].str().c_str());
          LOG_DEBUG("get model path", K(onnx_model_path));
        } else {
          onnx_model_path=ObString(match[0].str().c_str());
          LOG_DEBUG("get model path", K(onnx_model_path));
        }
    } else {
        LOG_DEBUG("get no model path");
    }
  return ret;
}

int ObTransformPyUDFMerge::extract_python_udf_expr_in_condition(
  ObIArray<ObPythonUdfRawExpr *> &python_udf_expr_list,
  ObIArray<ObRawExpr *> &src_exprs,
  ObIArray<oceanbase::share::schema::ObPythonUDFMeta > &python_udf_meta_list)
{
  int ret = OB_SUCCESS;
  for(int i=0;i<src_exprs.count();i++){
    if(OB_FAIL(ObTransformUtils::extract_all_python_udf_raw_expr_in_raw_expr(python_udf_expr_list,src_exprs.at(i)))){
      LOG_WARN("extract_all_python_udf_raw_expr_in_raw_expr fail", K(ret));
    }
  }

  for(int i=0;i<python_udf_expr_list.count();i++){
    python_udf_meta_list.push_back(python_udf_expr_list.at(i)->get_udf_meta());
  }
  return ret;
}

int ObTransformPyUDFMerge::merge_python_udf_expr_in_condition(
  ObIArray<int64_t> &idx_list,
  ObIArray<ObRawExpr *> &src_exprs)
{
  int ret = OB_SUCCESS;
  // test code
  int count=0;
  for (int i=1;i<idx_list.count();i++){
    src_exprs.remove((idx_list.at(i)-count));
    count++;
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

  if(pyudf_count>1) {
    need_trans=true;
    LOG_TRACE("need transform of ObTransformPyUDFMerge.", K(pyudf_count),K(ret));
  }
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
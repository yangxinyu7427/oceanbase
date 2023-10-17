#ifndef _OB_EXPR_PYTHON_UDF_
#define _OB_EXPR_PYTHON_UDF_

#include "sql/engine/expr/ob_expr_operator.h"
#include "sql/engine/python_udf_engine/python_udf_engine.h"

namespace  oceanbase {
namespace  sql {
class  ObExprPythonUdf : public  ObExprOperator {
public:
  explicit  ObExprPythonUdf(common::ObIAllocator &alloc);
  virtual  ~ObExprPythonUdf();

  virtual int calc_result_typeN(ObExprResType &type,
                                ObExprResType *types_array,
                                int64_t param_num,
                                common::ObExprTypeCtx &type_ctx) const;
  virtual int calc_resultN(common::ObObj &result,
                           const common::ObObj *objs,
                           int64_t param_num,
                           common::ObExprCtx &expr_ctx) const;
  
  virtual int cg_expr(ObExprCGCtx &op_cg_ctx,
                      const ObRawExpr &raw_expr,
                      ObExpr &rt_expr) const  override;

public:
  int set_udf_meta(const share::schema::ObPythonUDFMeta &udf);

  static int deep_copy_udf_meta(share::schema::ObPythonUDFMeta &dst,
                                common::ObIAllocator &alloc,
                                const share::schema::ObPythonUDFMeta &src);
                                
  int init_udf();

  static int eval_python_udf(const ObExpr &expr,
                             ObEvalCtx &ctx,
                             ObDatum &res);

  static int eval_python_udf_batch(const ObExpr &expr, ObEvalCtx &ctx,
                                   const ObBitVector &skip, const int64_t batch_size);

  static int eval_python_udf_batch_buffer(const ObExpr &expr, ObEvalCtx &ctx,
                                          const ObBitVector &skip, const int64_t batch_size);

  static int get_python_udf(pythonUdf* &pyudf, const ObExpr& expr);

  int ObExprPythonUdf::eval_test_udf(const ObExpr& expr, ObEvalCtx& ctx, ObDatum&  expr_datum)

  static int eval_test_udf(const ObExpr &expr,
                           ObEvalCtx &ctx,
                           ObDatum &res);

protected:
  common::ObIAllocator &allocator_;
  share::schema::ObPythonUDFMeta udf_meta_;
  ObSqlExpressionFactory sql_expression_factory_;
  ObExprOperatorFactory expr_op_factory_;
};

struct ObPythonUdfInfo : public ObIExprExtraInfo
{
  OB_UNIS_VERSION(1);
public:
  ObPythonUdfInfo(common::ObIAllocator &alloc, ObExprOperatorType type)
      : ObIExprExtraInfo(alloc, type), allocator_(alloc), udf_meta_() {}

  virtual int deep_copy(common::ObIAllocator &allocator,
                        const ObExprOperatorType type,
                        ObIExprExtraInfo *&copied_info) const override;

  int from_raw_expr(ObPythonUdfRawExpr &expr);

  common::ObIAllocator &allocator_;
  share::schema::ObPythonUDFMeta udf_meta_;
};

int ObPythonUdfInfo::deep_copy(common::ObIAllocator &allocator,
                                const ObExprOperatorType type,
                                ObIExprExtraInfo *&copied_info) const
{
  int ret = common::OB_SUCCESS;
  OZ(ObExprExtraInfoFactory::alloc(allocator, type, copied_info));
  ObPythonUdfInfo &other = *static_cast<ObPythonUdfInfo *>(copied_info);
  OZ(ObPythonUdfInfo::deep_copy_udf_meta(other.udf_meta_, allocator, udf_meta_));
  return ret;
}

int ObPythonUdfInfo::from_raw_expr(ObPythonUdfRawExpr &raw_expr)
{
  int ret = OB_SUCCESS;
  OZ(ObExprPythonUdf::deep_copy_udf_meta(udf_meta_, allocator_, raw_expr.get_udf_meta()));
  return ret;
}
} /* namespace sql */
} /* namespace oceanbase */

#endif
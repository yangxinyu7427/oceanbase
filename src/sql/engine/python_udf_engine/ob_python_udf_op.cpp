#define USING_LOG_PREFIX SQL_ENG

#include "ob_python_udf_op.h"
#include "sql/engine/ob_physical_plan.h"
#include "sql/engine/ob_exec_context.h"

namespace oceanbase
{
using namespace common;
namespace sql
{

OB_SERIALIZE_MEMBER((ObPythonUDFSpec, ObOpSpec),
                    udf_exprs_,
                    input_exprs_);

ObPythonUDFSpec::ObPythonUDFSpec(ObIAllocator &alloc, const ObPhyOperatorType type)
    : ObOpSpec(alloc, type), udf_exprs_(alloc), input_exprs_(alloc) {}

ObPythonUDFSpec::~ObPythonUDFSpec() {}

ObPythonUDFOp::ObPythonUDFOp(
    ObExecContext &exec_ctx, const ObOpSpec &spec, ObOpInput *input)
  : ObOperator(exec_ctx, spec, input), buf_alloc_(), tmp_alloc_(), controller_(buf_alloc_, tmp_alloc_)
{
  int ret = OB_SUCCESS;
  //predict_size_ = 256;
  //predict_size_ = MY_SPEC.max_batch_size_; //default

  const uint64_t tenant_id = ctx_.get_my_session()->get_effective_tenant_id();
  ObMemAttr attr(tenant_id, ObModIds::OB_SQL_EXPR, ObCtxIds::WORK_AREA);
  buf_alloc_.set_attr(attr);

  tmp_alloc_.set_tenant_id(tenant_id);
  tmp_alloc_.set_label(ObModIds::OB_SQL_EXPR);
  tmp_alloc_.set_ctx_id(ObCtxIds::WORK_AREA);
}

ObPythonUDFOp::~ObPythonUDFOp() {}

int ObPythonUDFOp::inner_open()
{
  int ret = OB_SUCCESS;
  // init context
  const uint64_t tenant_id = ctx_.get_my_session()->get_effective_tenant_id();
  if (OB_FAIL(controller_.init(MY_SPEC.max_batch_size_,
                               MY_SPEC.udf_exprs_, 
                               MY_SPEC.input_exprs_))) {
    LOG_WARN("Init python udf store controller failed", K(ret));
  } else {
    // check attrs
    ret = ObOperator::inner_open();
  }
  return ret;
}

int ObPythonUDFOp::inner_close()
{
  int ret = OB_SUCCESS;
  // free attrs;
  if (OB_FAIL(controller_.free())) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("Free python udf store controller failed", K(ret));
  } else {
    ret = ObOperator::inner_close();
  }
  return ret;
}

int ObPythonUDFOp::inner_rescan()
{
  int ret = OB_SUCCESS;
  // reset attrs
  const uint64_t tenant_id = ctx_.get_my_session()->get_effective_tenant_id();
  if (OB_FAIL(controller_.free()) || 
      OB_FAIL(controller_.init(MY_SPEC.max_batch_size_,
                               MY_SPEC.udf_exprs_, 
                               MY_SPEC.input_exprs_))) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    ret = ObOperator::inner_rescan();
  }
  return ret;
}

void ObPythonUDFOp::destroy()
{
  // destroy attrs
  ObOperator::destroy();
}

int ObPythonUDFOp::inner_get_next_row() 
{
  int ret = OB_SUCCESS;
  // not support yet
  clear_evaluated_flag();
  if (OB_FAIL(child_->get_next_row())) {
    if (OB_ITER_END != ret) {
      LOG_WARN("get row from child failed", K(ret));
    }
  } else {/* do nothing */}
  return ret;
}

/*使用缓存，确保每轮迭代能够以合适的批次大小计算Python UDF
* 1. 调用Python UDF expr的子expr
* 2. 复制frame上的输入至input缓存
* 3. 重复1，2至input缓存到达阈值
* 4. 直接在input缓存上计算Python UDF，并将结果保存至output缓存
* 5. 将output缓存复制回frame上的输出
* 6. 重复5至output缓存清空
 */
int ObPythonUDFOp::inner_get_next_batch(const int64_t max_row_cnt)
{
  int ret = OB_SUCCESS;
  // get data from buffer
  if (!controller_.is_output()) {
    clear_evaluated_flag();
    controller_.resize(controller_.get_desirable());
    tmp_alloc_.reset(); // clear temp data
    const ObBatchRows *child_brs = nullptr;
    while (OB_SUCC(ret) && !brs_.end_ && !controller_.is_full()) {
      if (OB_FAIL(child_->get_next_batch(max_row_cnt, child_brs))) {
        LOG_WARN("Get child next batch failed.", K(ret));
      } else if (brs_.copy(child_brs)) {
        LOG_WARN("Copy child batch rows failed.", K(ret));
      } else if (OB_FAIL(controller_.store(eval_ctx_, brs_))){
        LOG_WARN("Save input batchrows failed.", K(ret));
      }
    }
    if (OB_FAIL(ret) || OB_FAIL(controller_.process())) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("Process python udf failed.", K(ret));
    }
  }
  if (OB_FAIL(ret) || OB_FAIL(controller_.restore(eval_ctx_, brs_, max_row_cnt))) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("Restore output batchrows failed.", K(ret));
  }
  return ret;
}

/* find predict_size of buffer */
int ObPythonUDFOp::find_predict_size(ObExpr *expr, int32_t &predict_size)
{
  int ret = OB_SUCCESS;
  /*int expr_size = 256; //default
  if (expr->type_ == T_FUN_PYTHON_UDF) {
    int size = static_cast<ObPythonUdfInfo *>(expr->extra_info_)->predict_size;
    expr_size = size > expr_size ? size : expr_size;
  } else {
    for (int32_t i = 0; i < expr->arg_cnt_; i++) {
      find_predict_size(expr->args_[i], expr_size);
    }
  }
  if (predict_size < expr_size)
    predict_size = expr_size;
  if (predict_size > max_buffer_size_)
    predict_size = max_buffer_size_;*/
  return ret;
}

int ObPythonUDFOp::clear_calc_exprs_evaluated_flags() 
{
  int ret = OB_SUCCESS;
  for (int i = 0; i < MY_SPEC.calc_exprs_.count(); i++) {
    MY_SPEC.calc_exprs_[i]->get_eval_info(eval_ctx_).clear_evaluated_flag();
  }
  return ret;
}

/* ---------------------------------- ObColInputStore -------------------------------------*/
int ObColInputStore::init(const common::ObIArray<ObExpr *> &exprs,
                          int64_t batch_size,
                          int64_t length)
{
  int ret = OB_SUCCESS;
  exprs_.init(exprs.count());
  datums_copy_.init(exprs.count());
  for (int i = 0; OB_SUCC(ret) && i < exprs.count(); ++i) {
    ObExpr *expr = exprs.at(i);
    ObDatum *datums_buf = NULL;
    if (OB_ISNULL(datums_buf = static_cast<ObDatum *>(buf_alloc_.alloc(sizeof(ObDatum) * length)))) {
      //分配datums_buf空间
      ret = OB_INIT_FAIL;
      LOG_WARN("Memory allocation failed.", K(ret));
    } else if (OB_FAIL(exprs_.push_back(expr)) || OB_FAIL(datums_copy_.push_back(datums_buf))) {
      ret = OB_INIT_FAIL;
      LOG_WARN("Init datums_copy and other exprs failed.", K(ret));
    }
  }
  if (OB_SUCC(ret)) {
    length_ = length;
    batch_size_ = batch_size;
    saved_size_ = 0;
    output_idx_ = 0;
    inited_ = true;
  }
  return ret;
}

int ObColInputStore::free()
{
  int ret = OB_SUCCESS;
  datums_copy_.destroy();
  tmp_alloc_.reset();
  return ret;
}

int ObColInputStore::reuse()
{
  int ret = OB_SUCCESS;
  saved_size_ = 0;
  output_idx_ = 0;
  return ret;
}

int ObColInputStore::reset(int64_t length)
{
  int ret = OB_SUCCESS;
  if (length <= length_) {
    reuse();
  } else if (OB_FAIL(free())) {
  } else {
    datums_copy_.init(exprs_.count());
    for (int i = 0; i < exprs_.count(); ++i) {
      ObExpr *e = exprs_.at(i);
      ObDatum *datums_buf = static_cast<ObDatum *>(buf_alloc_.alloc(length * sizeof(ObDatum)));
      datums_copy_.push_back(datums_buf);
    }
    length_ = length;
    saved_size_ = 0;
    output_idx_ = 0;
  }
  return ret;
}

int ObColInputStore::save_batch(ObEvalCtx &eval_ctx, ObBatchRows &brs)
{
  int ret = OB_SUCCESS;
  int cnt = brs.size_ - brs.skip_->accumulate_bit_cnt(brs.size_);
  if (saved_size_ + cnt > length_) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("Other Input Store overflow.", K(ret));
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < exprs_.count(); ++i) {
      ObExpr *expr = exprs_.at(i);
      expr->cast_to_uniform(brs.size_, eval_ctx); // 数据格式转换过程可能会产生额外开销
      ObDatum *src = expr->locate_batch_datums(eval_ctx);
      ObDatum *dst = datums_copy_.at(i);
      int j = 0, zero = 0;
      int *src_idx;
      if (expr->is_const_expr())
        src_idx = &zero;
      else
        src_idx = &j;
      int dst_idx = saved_size_; // use dst idx to reform data
      for (j = 0; OB_SUCC(ret) && j < brs.size_; ++j) {
        if (brs.skip_->at(j)) {
          /* do nothing */
        } else {
          dst[dst_idx++].deep_copy(src[*src_idx], tmp_alloc_);
        }
      }
    }
    if (OB_SUCC(ret)) {
      saved_size_ += cnt;
    }
  }
  return ret;
}

int ObColInputStore::save_vector(ObEvalCtx &eval_ctx, ObBatchRows &brs)
{
  int ret = OB_SUCCESS;
  int cnt = brs.size_ - brs.skip_->accumulate_bit_cnt(brs.size_);
  if (saved_size_ + cnt > length_) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("Other Input Store overflow.", K(ret));
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < exprs_.count(); ++i) {
      ObExpr *expr = exprs_.at(i);
      ObIVector *vector = expr->get_vector(eval_ctx);
      ObDatum *dst = datums_copy_.at(i);
      int dst_idx = saved_size_; // use dst idx to reform data
      for (int64_t j = 0; OB_SUCC(ret) && j < brs.size_; ++j) {
        if (brs.skip_->at(j)) {
          /* do nothing */
        } else {
          // deep copy from vector to datums
          ObLength copy_len;
          const char *payload;
          vector->get_payload(j, payload, copy_len);
          char *buf = static_cast<char *>(tmp_alloc_.alloc(copy_len));
          if (OB_ISNULL(buf)) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
            LOG_WARN("allocate memory failed", K(copy_len), K(ret));
          } else {
            MEMCPY(buf, payload, copy_len);
            dst[dst_idx].ptr_ = buf;
            dst[dst_idx].len_ = copy_len;
            ++dst_idx;
          }
        }
      }
    }
    if (OB_SUCC(ret)) {
      saved_size_ += cnt;
    }
  }
  return ret;
}

int ObColInputStore::load_batch(ObEvalCtx &eval_ctx, int64_t load_size)
{
  int ret = OB_SUCCESS;
  if (output_idx_ + load_size > saved_size_) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("Not enough data to load.", K(ret));
  } else {
    for (int64_t i = 0; i < exprs_.count(); ++i) {
      ObExpr *expr = exprs_.at(i);
      ObDatum *src = datums_copy_.at(i);
      ObDatum *dst = expr->locate_batch_datums(eval_ctx);
      for (int j = 0; j < load_size; ++j) {
        dst[j].set_datum(src[output_idx_ + j]);
      }
    }
    if (OB_SUCC(ret)) {
      output_idx_ += load_size;
    }
  }
  return ret;
}

int ObColInputStore::load_vector(ObEvalCtx &eval_ctx, int64_t load_size)
{
  int ret = OB_SUCCESS;
  if (output_idx_ + load_size > saved_size_) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("Not enough data to load.", K(ret));
  } else {
    for (int64_t i = 0; i < exprs_.count(); ++i) {
      ObExpr *expr = exprs_.at(i);
      ObDatum *src = datums_copy_.at(i);
      ObIVector *dst = expr->get_vector(eval_ctx);
      for (int j = 0; j < load_size; ++j) {
        dst->set_payload(j, src[output_idx_ + j].ptr_, src[output_idx_ + j].len_);
      }
    }
    if (OB_SUCC(ret)) {
      output_idx_ += load_size;
    }
  }
  return ret;
}
/* ---------------------------------- ObPUInputStore -------------------------------- */
int ObPUInputStore::init(common::ObIAllocator *alloc, ObExpr *expr, int64_t length)
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(inited_)) {
    ret = OB_INIT_TWICE;
    LOG_WARN("Repeat allocation", K(ret));
  } else if (OB_ISNULL(alloc) || OB_ISNULL(expr) || length <= 0) {
    ret = OB_NOT_INIT;
    LOG_WARN("Uninit allocator and expression.", K(ret));
  } else if (expr->type_ != T_FUN_PYTHON_UDF) {
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("Not Python UDF Expr Type.", K(ret));
  } else {
    alloc_ = alloc;
    expr_ = expr;
    length_ = length;
    if (OB_FAIL(alloc_data_ptrs()) || OB_ISNULL(data_ptrs_)) {
      ret = OB_INIT_FAIL;
      LOG_WARN("Init ObPUInputStore failed.", K(ret), K(alloc), K(expr->arg_cnt_));
    } else {
      inited_ = true;
      saved_size_ = 0;
    }
  }
  return ret;
}

int ObPUInputStore::alloc_data_ptrs()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(inited_) ) {
    ret = OB_INIT_TWICE;
    LOG_WARN("Repeat allocation.", K(ret));
  } else if (OB_ISNULL(alloc_) || OB_ISNULL(expr_) || expr_->arg_cnt_ <= 0 || length_ <= 0) {
    ret = OB_NOT_INIT;
    LOG_WARN("Uninit allocator and expression.", K(ret));
  } else {
    data_ptrs_ = static_cast<char **>(alloc_->alloc(expr_->arg_cnt_ * sizeof(char *))); // data lists
    if (OB_ISNULL(data_ptrs_)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("Allocate data_ptr_ memory failed", K(ret), K(alloc_), K(expr_->arg_cnt_));
    }
    for (int i = 0; i < expr_->arg_cnt_; ++i) {
      ObExpr *e = expr_->args_[i];
      /* allocate by datum type */
      switch(e->datum_meta_.type_) {
        case ObCharType:
        case ObVarcharType:
        case ObTinyTextType:
        case ObTextType:
        case ObMediumTextType:
        case ObLongTextType: {
          data_ptrs_[i] = reinterpret_cast<char *>(alloc_->alloc(length_ * sizeof(char *))); // string lists
          break;
        }
        case ObTinyIntType:
        case ObSmallIntType:
        case ObMediumIntType:
        case ObInt32Type:
        case ObIntType: {
          data_ptrs_[i] = reinterpret_cast<char *>(alloc_->alloc(length_ * sizeof(int))); // int lists
          break;
        }
        case ObDoubleType: {
          data_ptrs_[i] = reinterpret_cast<char *>(alloc_->alloc(length_ * sizeof(double))); // double lists
          break;
        }
        default: {
          ret = OB_NOT_SUPPORTED;
          LOG_WARN("Unsupported input arg type, alloc_data_ptrs failed.", K(ret));
        }
      }
      if (OB_ISNULL(data_ptrs_[i])) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("Allocate data_ptr_[i] memory failed.", K(ret), K(alloc_), K(length_), K(i), K(e->datum_meta_.type_));
      }
    }
  }
  return ret;
}

int ObPUInputStore::reuse()
{
  int ret = OB_SUCCESS;
  saved_size_ = 0;
  return ret;
}

int ObPUInputStore::reset(int64_t length /* default = 0 */)
{
  int ret = OB_SUCCESS;
  if(length <= length_) {
    ret = reuse();
  } else if (OB_FAIL(free())) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(alloc_data_ptrs())) {
    ret = OB_INIT_FAIL;
  } else {
    saved_size_ = 0;
    inited_ = true;
  }
  return ret;
}

int ObPUInputStore::free() {
  int ret = OB_SUCCESS;
  // malloc allocator支持free(ptr)
  // if expr_ = null, do not check
  for (int i = 0; OB_SUCC(ret) && i < expr_->arg_cnt_; ++i) {
    ObExpr *e = expr_->args_[i];
    /* allocate by datum type */
    switch(e->datum_meta_.type_) {
      case ObCharType:
      case ObVarcharType:
      case ObTinyTextType:
      case ObTextType:
      case ObMediumTextType:
      case ObLongTextType: {
        alloc_->free(reinterpret_cast<char **>(data_ptrs_[i]));
        break;
      }
      case ObTinyIntType:
      case ObSmallIntType:
      case ObMediumIntType:
      case ObInt32Type:
      case ObIntType: {
        alloc_->free(reinterpret_cast<int *>(data_ptrs_[i]));
        break;
      }
      case ObDoubleType: {
        alloc_->free(reinterpret_cast<double *>(data_ptrs_[i]));
        break;
      }
      default: {
        ret = OB_NOT_SUPPORTED;
        LOG_WARN("Unsupported input arg type, free failed in ObPUDataStore.", K(ret));
      }
    }
  }
  return ret;
}

int ObPUInputStore::save_batch(ObEvalCtx &eval_ctx, ObBatchRows &brs)
{
  int ret = OB_SUCCESS;
  int brs_cnt = brs.size_ - brs.skip_->accumulate_bit_cnt(brs.size_);
  if (OB_ISNULL(expr_) || expr_->arg_cnt_ <= 0) {
    // empty exprs_: do nothing
  } else if (OB_ISNULL(data_ptrs_) || !inited_) {
    ret = OB_NOT_INIT;
  } else if (saved_size_ + brs_cnt <= length_) {
    for (int64_t i = 0; i < expr_->arg_cnt_; i++) {
      ObExpr *e = expr_->args_[i];
      ObDatum *src = e->locate_batch_datums(eval_ctx);
      int j = 0, zero = 0;
      int64_t dst_idx = saved_size_;
      int *src_idx;
      if (!expr_->args_[i]->is_const_expr()) 
        src_idx = &j;
      else 
        src_idx = &zero;
      switch(e->datum_meta_.type_) {
        case ObCharType:
        case ObVarcharType:
        case ObTinyTextType:
        case ObTextType:
        case ObMediumTextType:
        case ObLongTextType: {
          PyObject **dst = reinterpret_cast<PyObject **>(data_ptrs_[i]);
          for (j = 0; j < brs.size_; j++) {
            if (!brs.skip_->at(j)) {
              // copy string into pyobject
              ObDatum &src_datum = src[*src_idx];
              PyObject *buf = PyUnicode_FromStringAndSize(src_datum.ptr_, src_datum.len_);
              dst[dst_idx++] = buf;
            }
          }
          break;
        }
        case ObTinyIntType:
        case ObSmallIntType:
        case ObMediumIntType:
        case ObInt32Type:
        case ObIntType: {
          int *dst = reinterpret_cast<int *>(data_ptrs_[i]);
          for (j = 0; j < brs.size_; j++) {
            ObDatum &src_datum = src[*src_idx];
            if (!brs.skip_->at(j)) {
              // copy int
              dst[dst_idx++] = src_datum.get_int();
            }
          }
          break;
        }
        case ObDoubleType: {
          double *dst = reinterpret_cast<double *>(data_ptrs_[i]);
          for (j = 0; j < brs.size_; j++) {
            ObDatum &src_datum = src[*src_idx];
            if (!brs.skip_->at(j)) {
              // copy double
              dst[dst_idx++] = src_datum.get_double();
            }
          }
          break;
        }
        default: {
          ret = OB_NOT_SUPPORTED;
          LOG_WARN("Unsupported input arg type, save data failed in ObPUDataStore.", K(ret));
        }
      }
      if (OB_ISNULL(data_ptrs_[i])) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("Allocate data_ptr_[i] memory failed.", K(ret), K(alloc_), K(length_), K(i), K(e->datum_meta_.type_));
      }
    }
    saved_size_ += brs_cnt;
  } else {
    // op逻辑控制其不能超过阈值
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ObPUDataStore Stack Overflow.", K(ret));
  }
  return ret;
}

int ObPUInputStore::save_vector(ObEvalCtx &eval_ctx, ObBatchRows &brs)
{
  // not implemented yet
  int ret = OB_NOT_SUPPORTED;
  UNUSED(eval_ctx);
  UNUSED(brs);
  return ret;
}

/* ----------------------------------- ObPythonUDFCell --------------------------------- */
int ObPythonUDFCell::init(common::ObIAllocator *buf_alloc, 
                          common::ObIAllocator *tmp_alloc, 
                          ObExpr *expr, 
                          int64_t batch_size, 
                          int64_t length)
{
  int ret = OB_SUCCESS;
  // init python udf expr according to its metadata
  if (OB_FAIL(input_store_.init(buf_alloc, expr, length))) {
    ret = OB_NOT_INIT;
    LOG_WARN("Init input store failed.", K(ret));
  } else {
    buf_alloc_ = buf_alloc;
    tmp_alloc_ = tmp_alloc;
    expr_ = expr;
    //desirable_ = expr->extra_info_.;
    ObPythonUdfInfo *info = static_cast<ObPythonUdfInfo *>(expr_->extra_info_);
    desirable_ = info->predict_size; // 初始值256
    batch_size_ = batch_size;
  }
  return ret;
}

int ObPythonUDFCell::free()
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(input_store_.free())) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("Free Python UDF cell failed.", K(ret));
  }
  return ret;
}

int ObPythonUDFCell::do_store(ObEvalCtx &eval_ctx, ObBatchRows &brs)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(expr_)) {
    ret = OB_NOT_INIT;
    LOG_WARN("Expr in input store is NULL.", K(ret));
  } else {
    // decide save function according to data transfer type
    // 因为要判断skip并进行重整，不能直接MEMCPY，所以用save_batch方法就足够了
    ObBitVector &my_skip = expr_->get_pvt_skip(eval_ctx);
    for (int64_t i = 0; i < expr_->arg_cnt_ && OB_SUCC(ret); i++) {
      ObExpr *e = expr_->args_[i];
      // eval_batch() = eval_vector() + cast_to_uniform()
      if (OB_FAIL(e->eval_batch(eval_ctx, my_skip, batch_size_))) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("Eval Python UDF args' vector/batch result failed.", K(ret));
      }
    }

    // expr参数结果已经被设置为UNIFORM格式
    if (OB_FAIL(ret) || OB_FAIL(input_store_.save_batch(eval_ctx, brs))) { 
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("Save Python UDF input batch failed.", K(ret), K(expr_), K(batch_size_));
    }
  }
  return ret;
}

int ObPythonUDFCell::do_process_all()
{
  int ret = OB_SUCCESS;
  // pre process
  result_store_ = NULL;
  result_size_ = 0;
  //Ensure GIL
  bool nStatus = PyGILState_Check();
  PyGILState_STATE gstate;
  if(!nStatus) {
    gstate = PyGILState_Ensure();
    nStatus = true;
  }
  //load numpy api
  _import_array();
    PyObject *pArgs = NULL;
    int64_t eval_size;
    if (OB_FAIL(wrap_input_numpy(pArgs, eval_size)) || OB_ISNULL(pArgs)) { // wrap all input
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("Wrap Cell Input Store as Python UDF input args failed.", K(ret));
    } else if (OB_FAIL(eval(pArgs, eval_size))) { // evaluation and keep the result
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("Eval Python UDF failed.", K(ret));
    } else { /* do nothing */ }
  
  // Release GIL
  if(nStatus)
    PyGILState_Release(gstate);

  if (OB_SUCC(ret)) {
    input_store_.reuse();
  }
  return ret;
}

int ObPythonUDFCell::do_process()
{
  int ret = OB_SUCCESS;
  // pre process
  result_store_ = NULL;
  result_size_ = 0;
  //Ensure GIL
  bool nStatus = PyGILState_Check();
  PyGILState_STATE gstate;
  if(!nStatus) {
    gstate = PyGILState_Ensure();
    nStatus = true;
  }
  //load numpy api
  _import_array();
  struct timeval t1, t2;
  int64_t eval_size = 0;
  for (int idx = 0; OB_SUCC(ret) && idx < input_store_.get_saved_size(); idx += eval_size) {
    gettimeofday(&t1, NULL);
    PyObject *pArgs = NULL;
    if (OB_FAIL(wrap_input_numpy(pArgs, idx, desirable_, eval_size)) || OB_ISNULL(pArgs)) { // wrap the input
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("Wrap Cell Input Store as Python UDF input args failed.", K(ret));
    } else if (OB_FAIL(eval(pArgs, eval_size))) { // evaluation and keep the result
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("Eval Python UDF failed.", K(ret));
    } else {
      gettimeofday(&t2, NULL);
      modify_desirable(t1, t2, eval_size);
    }
  }
  // Release GIL
  if(nStatus)
    PyGILState_Release(gstate);

  return ret;
}

int ObPythonUDFCell::do_restore(ObEvalCtx &eval_ctx, int64_t output_idx, int64_t output_size)
{
  int ret = OB_SUCCESS;
  if (output_idx + output_size > result_size_) { //确保访问结果数组时不会越界
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("Result Store Overflow.", K(ret));
  } else if (OB_ISNULL(expr_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("Unexpected NULL expr_.", K(ret));
  } else if (expr_->enable_rich_format() && OB_FAIL(do_restore_vector(eval_ctx, output_idx, output_size))){
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("Unsupported result type.", K(ret));
  } else if (!expr_->enable_rich_format() && OB_FAIL(do_restore_batch(eval_ctx, output_idx, output_size))) {
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("Unsupported result type.", K(ret));
  } else {
    expr_->set_evaluated_projected(eval_ctx);
  }
  return ret;
}

int ObPythonUDFCell::do_restore_batch(ObEvalCtx &eval_ctx, int64_t output_idx, int64_t output_size)
{
  int ret = OB_SUCCESS;
  ObDatum *result_datums = expr_->locate_batch_datums(eval_ctx);
  PyArrayObject *result_store = reinterpret_cast<PyArrayObject *>(result_store_);
  switch(expr_->datum_meta_.type_) {
    case ObCharType:
    case ObVarcharType:
    case ObTinyTextType:
    case ObTextType:
    case ObMediumTextType:
    case ObLongTextType: {
      for (int i = 0; i < output_size; ++i) {
        expr_->reset_ptr_in_datum(eval_ctx, i);
        result_datums[i].set_string(common::ObString(PyUnicode_AsUTF8(
          PyArray_GETITEM(result_store, (char *)PyArray_GETPTR1(result_store, output_idx + i)))));
      }
      break;
    }
    case ObTinyIntType:
    case ObSmallIntType:
    case ObMediumIntType:
    case ObInt32Type:
    case ObIntType: {
      for (int i = 0; i < output_size; ++i) {
        expr_->reset_ptr_in_datum(eval_ctx, i);
        result_datums[i].set_int(PyLong_AsLong(
          PyArray_GETITEM(result_store, (char *)PyArray_GETPTR1(result_store, output_idx + i))));
      }
      break;
    }
    case ObDoubleType: {
      for (int i = 0; i < output_size; ++i) {
        expr_->reset_ptr_in_datum(eval_ctx, i);
        result_datums[i].set_double(PyFloat_AsDouble(
          PyArray_GETITEM(result_store, (char *)PyArray_GETPTR1(result_store, output_idx + i))));
      }
      break;
    }
    default: {
      //error
      ret = OB_NOT_SUPPORTED;
      LOG_WARN("Unsupported result type.", K(ret));
    }
  }
  return ret;
}

int ObPythonUDFCell::do_restore_vector(ObEvalCtx &eval_ctx, int64_t output_idx, int64_t output_size)
{
  int ret = OB_SUCCESS;
  if (!expr_->get_eval_info(eval_ctx).evaluated_) {
    VectorFormat format = VEC_INVALID;
    const ObPythonUdfInfo *info = static_cast<ObPythonUdfInfo *>(expr_->extra_info_);
    switch(info->udf_meta_.ret_) {
      case share::schema::ObPythonUDF::PyUdfRetType::STRING:
        //format = VEC_DISCRETE;
        format = VEC_CONTINUOUS;
      break;
      case share::schema::ObPythonUDF::PyUdfRetType::INTEGER:
      case share::schema::ObPythonUDF::PyUdfRetType::REAL:
        format = VEC_FIXED;
      break;
      default:
        ret = OB_NOT_SUPPORTED;
        LOG_WARN("Unsupported result type.", K(ret));
    }
    expr_->init_vector_for_write(eval_ctx, format, batch_size_);
  }
  ObIVector *vector = expr_->get_vector(eval_ctx);
  PyArrayObject *result_store = reinterpret_cast<PyArrayObject *>(result_store_);
  // 构造vector并赋回expr_
  switch(expr_->datum_meta_.type_) {
    case ObCharType:
    case ObVarcharType:
    case ObTinyTextType:
    case ObTextType:
    case ObMediumTextType:
    case ObLongTextType: {
      for (int i = 0; i < output_size; ++i) {
        vector->set_string(i, common::ObString(PyUnicode_AsUTF8(
          PyArray_GETITEM(result_store, (char *)PyArray_GETPTR1(result_store, output_idx + i)))));
      }
      break;
    }
    case ObTinyIntType:
    case ObSmallIntType:
    case ObMediumIntType:
    case ObInt32Type:
    case ObIntType: {
      //VectorFormat format = VEC_FIXED;
      //expr_->init_vector_for_write(eval_ctx, format, batch_size_);
      for (int i = 0; i < output_size; ++i) {
        vector->set_int(i, PyLong_AsLong(
          PyArray_GETITEM(result_store, (char *)PyArray_GETPTR1(result_store, output_idx + i))));
      }
      break;
    }
    case ObDoubleType: {
      for (int i = 0; i < output_size; ++i) {
        vector->set_double(i, PyFloat_AsDouble(
          PyArray_GETITEM(result_store, (char *)PyArray_GETPTR1(result_store, output_idx + i))));
      }
      break;
    }
    default: {
      //error
      ret = OB_NOT_SUPPORTED;
      LOG_WARN("Unsupported result type.", K(ret));
    }
  }
  return ret;
}

// warp all saved input
int ObPythonUDFCell::wrap_input_numpy(PyObject *&pArgs, int64_t &eval_size)
{
  return wrap_input_numpy(pArgs, 0, input_store_.get_saved_size(), eval_size);
}

// warp [idx, idx + predict_size_]
int ObPythonUDFCell::wrap_input_numpy(PyObject *&pArgs, int64_t idx, int64_t predict_size, int64_t &eval_size)
{
  int ret = OB_SUCCESS;
  pArgs = PyTuple_New(expr_->arg_cnt_); // malloc hook
  int64_t saved_size = input_store_.get_saved_size();
  eval_size = (idx + predict_size) < saved_size ? predict_size : saved_size - idx;
  npy_intp elements[1] = {eval_size};
  if (OB_ISNULL(expr_)) {
    ret = OB_NOT_INIT;
    LOG_WARN("Expr in input store is NULL.", K(ret));
  } else {
    for (int i = 0; i < expr_->arg_cnt_; ++i) {
      PyObject *numpyarray = NULL;
      switch (expr_->args_[i]->datum_meta_.type_) {
        case ObCharType:
        case ObVarcharType:
        case ObTinyTextType:
        case ObTextType:
        case ObMediumTextType:
        case ObLongTextType: {
          /*numpyarray = PyArray_New(&PyArray_Type, 
                                  1, 
                                  elements, 
                                  NPY_OBJECT, 
                                  NULL, 
                                  reinterpret_cast<PyObject **>(input_store_.get_data_ptr_at(i)), 
                                  eval_size, 
                                  0, 
                                  NULL);*/
          numpyarray = PyArray_New(&PyArray_Type, 1, elements, NPY_OBJECT, NULL, NULL, 0, 0, NULL);
          PyObject **unicode_strs = reinterpret_cast<PyObject **>(input_store_.get_data_ptr_at(i)) + idx;
          for (int j = 0; j < eval_size; ++j) {
            //put unicode string pyobject into numpy array
            PyArray_SETITEM((PyArrayObject *)numpyarray, (char *)PyArray_GETPTR1((PyArrayObject *)numpyarray, j), 
              unicode_strs[j]);
          }
          break;
        }
        case ObTinyIntType:
        case ObSmallIntType:
        case ObMediumIntType:
        case ObInt32Type:
        case ObIntType: {
          numpyarray = PyArray_New(&PyArray_Type, 
                                  1, 
                                  elements, 
                                  NPY_INT32, 
                                  NULL, 
                                  reinterpret_cast<int *>(input_store_.get_data_ptr_at(i)) + idx,  
                                  eval_size, 
                                  0, 
                                  NULL);
          break;
        }
        case ObDoubleType: {
          numpyarray = PyArray_New(&PyArray_Type, 
                                  1, 
                                  elements, 
                                  NPY_FLOAT64, 
                                  NULL, 
                                  reinterpret_cast<double *>(input_store_.get_data_ptr_at(i)) + idx, 
                                  eval_size, 
                                  0, 
                                  NULL);
          break;
        }
        default: {
          //error
          ret = OB_NOT_SUPPORTED;
          LOG_WARN("Unsupported input type.", K(ret));
        }
      }
      if(PyTuple_SetItem(pArgs, i, numpyarray) != 0){
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("Set numpy array arg failed.", K(ret));
      }
    }
  }
  return ret;
}

int ObPythonUDFCell::eval(PyObject *pArgs, int64_t eval_size)
{
  int ret = OB_SUCCESS;
  // evalatuion in python interpreter

  // extract pyfun handler
  ObPythonUdfInfo *info = static_cast<ObPythonUdfInfo *>(expr_->extra_info_);
  std::string name(info->udf_meta_.name_.ptr());
  name = name.substr(0, info->udf_meta_.name_.length());
  std::string pyfun_handler = name.append("_pyfun");

  PyObject *pModule = NULL;
  PyObject *pFunc = NULL;
  PyObject *pResult = NULL;

  if (OB_ISNULL(pModule = PyImport_AddModule("__main__"))) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("Import main module failed.", K(ret));
  } else if (OB_ISNULL(pFunc = PyObject_GetAttrString(pModule, pyfun_handler.c_str())) 
      || !PyCallable_Check(pFunc)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("Get function handler failed.", K(ret));
  } else if (OB_ISNULL(pResult = PyObject_CallObject(pFunc, pArgs))) {
    ObExprPythonUdf::process_python_exception();
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("Execute Python UDF error.", K(ret));
  } else {
    // numpy array concat
    if (OB_ISNULL(result_store_)) {
      result_store_ = reinterpret_cast<void *>(pResult);
    } else {
      PyObject *concat = PyTuple_New(2);
      PyTuple_SetItem(concat, 0, reinterpret_cast<PyObject *>(result_store_));
      PyTuple_SetItem(concat, 1, pResult);
      result_store_ = reinterpret_cast<void *>(PyArray_Concatenate(concat, 0));
    }
    result_size_ += eval_size;
  }
  return ret;
}

int ObPythonUDFCell::modify_desirable(timeval &start, timeval &end, int64_t eval_size)
{
  int ret = OB_SUCCESS;
  ObPythonUdfInfo *info = static_cast<ObPythonUdfInfo *>(expr_->extra_info_);
  double timeuse = (end.tv_sec - start.tv_sec) * 1000000 + (double)(end.tv_usec - start.tv_usec); // usec
  double tps = eval_size * 1000000 / timeuse; // current tuples per sec
  if (info->tps_s == 0) { // 初始化
    info->tps_s = tps;
    info->predict_size += info->delta; // 尝试调整
  } else if (info->round > info->round_limit || eval_size != info->predict_size) { //超过轮次，停止调整batch size 或 不符合predict size
    // do nothing
  } else if (tps > info->tps_s) { 
    // 提升阈值λ为10% 且 目前计算数量与给定batch size相符，进行调整
    if (tps < (1 + info->lambda) * info->tps_s)
      info->round++;
    info->tps_s = tps;
    info->predict_size += info->delta;
  } else { //tps <= info->tps_s
    // 未达到阈值
    info->tps_s = (1 - info->alpha) * info->tps_s + info->alpha * tps; // 平滑系数α
    if (tps > (1 - info->lambda) * info->tps_s)
      info->round++;
  }
  if (OB_SUCC(ret)) {
    desirable_ = info->predict_size;
  }
  return ret;
}

/* ----------------------------------- ObPUStoreController --------------------------------- */
int ObPUStoreController::init(int64_t batch_size,
                              const common::ObIArray<ObExpr *> &udf_exprs, 
                              const common::ObIArray<ObExpr *> &input_exprs)
{
  int ret = OB_SUCCESS;
  // init python udf cells
  for (int i = 0; i < udf_exprs.count() && OB_SUCC(ret); ++i) {
    ObExpr *udf_expr = udf_exprs.at(i);
    void *cell_ptr = static_cast<ObPythonUDFCell *>(buf_alloc_.alloc(sizeof(ObPythonUDFCell))); // buffer alloc
    ObPythonUDFCell *cell = new (cell_ptr) ObPythonUDFCell();
    if (OB_ISNULL(cell) || OB_FAIL(cell->init(&buf_alloc_, &tmp_alloc_, udf_expr, batch_size, capacity_)) ) {
      ret = OB_INIT_FAIL;
      LOG_WARN("Init Python UDF Cell failed.", K(ret));
    } else {
      cells_list_.add_last(cell);
    }
  }

  if (OB_SUCC(ret)) {
    if (OB_FAIL(other_store_.init(input_exprs, batch_size, capacity_))) {
      ret = OB_INIT_FAIL;
      LOG_WARN("Init other input store failed.", K(ret));
    } else {
      stored_input_cnt_ = 0;
      stored_output_cnt_ = 0;
      output_idx_ = 0;
      batch_size_ = batch_size;
    }
  }
  return ret;
}

int ObPUStoreController::free()
{
  int ret = OB_SUCCESS;
  ObPythonUDFCell* header = cells_list_.get_header();
  for (ObPythonUDFCell* cell = cells_list_.get_first(); 
       cell != header && OB_SUCC(ret); 
       cell = cell->get_next()) {
    if (OB_FAIL(cell->free())) {
      ret = OB_NOT_SUPPORTED;
      LOG_WARN("Free Python UDF Cell failed.", K(ret));
      break;
    }
  }
  if (OB_SUCC(ret)) {
    if (OB_FAIL(other_store_.free())) {
      ret = OB_NOT_SUPPORTED;
      LOG_WARN("Free other input store failed.", K(ret));
    } else {
      // reset cnt & idx
      stored_input_cnt_ = 0;
      stored_output_cnt_ = 0;
      output_idx_ = 0;
    }
  }
  return ret;
}

int ObPUStoreController::store(ObEvalCtx &eval_ctx, ObBatchRows &brs)
{
  int ret = OB_SUCCESS;
  // do cells input store
  int64_t cnt = brs.size_ - brs.skip_->accumulate_bit_cnt(brs.size_);
  ObPythonUDFCell* header = cells_list_.get_header();
  for (ObPythonUDFCell* cell = cells_list_.get_first(); 
       cell != header && OB_SUCC(ret); 
       cell = cell->get_next()) {
    if (OB_FAIL(cell->do_store(eval_ctx, brs))) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("Do Python UDF Cell Store failed.", K(ret));
    }
  }
  if (OB_SUCC(ret)) {
    // do other inputs store
    // 要判断skip进行数据重整，使用local allocator进行数据复制
    // load时要转为相应类型的vector
    // 下层传上来的数据不一定为uniform格式，需要save_vector()或cast_to_uniform()
    if (OB_FAIL(other_store_.save_vector(eval_ctx, brs))) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("Save other input cols failed.", K(ret));
    } else {
      stored_input_cnt_ += cnt;
    }
  }
  return ret;
}

int ObPUStoreController::process()
{
  int ret = OB_SUCCESS;
  if (is_empty()) {
    /* do nothing */
  } else {
    ObPythonUDFCell* header = cells_list_.get_header();
    for (ObPythonUDFCell* cell = cells_list_.get_first(); 
        cell != header && OB_SUCC(ret); 
        cell = cell->get_next()) {
      if (cell->get_store_size() != stored_input_cnt_) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("Unsaved input rows.", K(ret));
      } else if (OB_FAIL(cell->do_process())) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("Do Python UDF Cell process udf failed.", K(ret));
      } else if (cell->get_result_size() != stored_input_cnt_){
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("Unprocessed input rows.", K(ret));
      } else {
        cell->reset_input_store();
      }
    }
    if (OB_SUCC(ret)) {
      stored_output_cnt_ = stored_input_cnt_;
      stored_input_cnt_ = 0;
      output_idx_ = 0;
    }
  }
  return ret;
}

int ObPUStoreController::restore(ObEvalCtx &eval_ctx, ObBatchRows &brs, int64_t max_row_cnt)
{
  int ret = OB_SUCCESS;
  if (stored_output_cnt_ <= 0 || !is_output()) { // Not enough output data
    /* do nothing */
  } else {
    int64_t output_size = batch_size_ < (stored_output_cnt_ - output_idx_) 
                        ? batch_size_ 
                        : (stored_output_cnt_ - output_idx_);
    ObPythonUDFCell* header = cells_list_.get_header();
    for (ObPythonUDFCell* cell = cells_list_.get_first(); 
        cell != header && OB_SUCC(ret); 
        cell = cell->get_next()) {
      if (OB_FAIL(cell->do_restore(eval_ctx, output_idx_, output_size))) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("Do Python UDF Cell restore failed.", K(ret));
      }
    }
    if (OB_SUCC(ret)) {
      if (OB_FAIL(other_store_.load_vector(eval_ctx, output_size))) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("Load other input cols failed.", K(ret));
      } else {
        output_idx_ += output_size;
        // reset batchrows
        brs.size_ = output_size;
        brs.reset_skip(output_size);
        brs.end_ = false;
      }
    }
    if (OB_SUCC(ret) && end_output()) {
      other_store_.reuse();
    }
  }
  return ret;
}

int ObPUStoreController::resize(int64_t size)
{
  int ret = OB_SUCCESS;
  if (size <= capacity_) {
    stored_input_cnt_ = 0;
    stored_output_cnt_ = 0;
    output_idx_ = 0;
  } else {
    ObPythonUDFCell* header = cells_list_.get_header();
    for (ObPythonUDFCell* cell = cells_list_.get_first(); 
        cell != header && OB_SUCC(ret); 
        cell = cell->get_next()) {
      if (OB_FAIL(cell->reset(size))) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("Do Python UDF Cell resize failed.", K(ret));
      }
    }
    if (OB_SUCC(ret)) {
      ret = other_store_.reset(size);
    }
  }
  return ret;
}

} // end namespace sql
} // end namespace oceanbase
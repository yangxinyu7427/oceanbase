#define USING_LOG_PREFIX SQL_ENG

#include "ob_python_udf_op.h"
#include "sql/engine/ob_physical_plan.h"
#include "sql/engine/ob_exec_context.h"

namespace oceanbase
{
using namespace common;
namespace sql
{

ObPythonUDFSpec::ObPythonUDFSpec(ObIAllocator &alloc, const ObPhyOperatorType type)
    : ObSubPlanScanSpec(alloc, type) {}

ObPythonUDFSpec::~ObPythonUDFSpec() {}

ObPythonUDFOp::ObPythonUDFOp(
    ObExecContext &exec_ctx, const ObOpSpec &spec, ObOpInput *input)
  : ObSubPlanScanOp(exec_ctx, spec, input)
{
  buffer_.init(MY_SPEC.projector_, exec_ctx, spec_.max_batch_size_);
  //initialize Python Intepreter
  //if(Py_IsInitialized())
    //Py_Finalize();
  //Py_Initialize();
  //PyEval_SaveThread();
  //Py_InitializeEx(!Py_IsInitialized());
  //_save = PyEval_SaveThread();
}

ObPythonUDFOp::~ObPythonUDFOp()
{
  //PyEval_RestoreThread((PyThreadState *)_save);
  //Py_FinalizeEx(); // Python Intepreter
}

int ObPythonUDFOp::inner_get_next_batch(const int64_t max_row_cnt)
{
  int ret = OB_SUCCESS;
  while (OB_SUCC(ret) && (buffer_.get_size() < buffer_.get_max_size() / 2) && !brs_.end_) {
    if (OB_FAIL(ObSubPlanScanOp::inner_get_next_batch(max_row_cnt))) {
      LOG_WARN("fail to inner get next batch", K(ret));
    } else if (OB_FAIL(buffer_.save(eval_ctx_, brs_))){
      LOG_WARN("fail to save batchrows", K(ret));
    }
  }
  if (OB_FAIL(buffer_.load(eval_ctx_, brs_, spec_.max_batch_size_))) {
    LOG_WARN("fail to load batchrows", K(ret));
  }
  return ret;
  //return ObSubPlanScanOp::inner_get_next_batch(max_row_cnt);
}

/* ------------------------------------ buffer for python_udf ----------------------------------- */
int ObVectorBuffer::init(const common::ObIArray<ObExpr *> &exprs, ObExecContext &exec_ctx, int64_t max_batch_size)
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(inited_)) {
    // do nothing
  } else if (exprs.count() == 0) {
    // do nothing
  } else {
    exprs_ = &exprs;
    exec_ctx_ = &exec_ctx;
    max_size_ = max_batch_size * 2; // default 8192, just use 4096?
    datums_ = static_cast<ObDatum *>(exec_ctx.get_allocator().alloc(
      max_size_ * exprs.count() * sizeof(ObDatum)));
    if (OB_ISNULL(datums_)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("allocate memory failed", K(ret), K(max_size_), K(exprs.count()));
    }
    inited_ = true;
    saved_size_ = 0;
  }
  return ret;
}

int ObVectorBuffer::save(ObEvalCtx &eval_ctx, ObBatchRows &brs_) 
{
  int ret = OB_SUCCESS;
  int cnt = brs_.size_ - brs_.skip_->accumulate_bit_cnt(brs_.size_);
  if (NULL == exprs_) {
    // empty expr_: do nothing
  } else if (NULL == datums_) {
    ret = OB_NOT_INIT;
  } else if (saved_size_ + cnt <= max_size_) {
    for (int64_t i = 0; i < exprs_->count(); i++) {
      ObExpr *e = exprs_->at(i);
      ObDatum *src = e->locate_batch_datums(eval_ctx);
      int64_t k = 0;
      for (int64_t j = 0; j < brs_.size_; j++) {
        /* remove skipped rows */
        if (!brs_.skip_->at(j)) {
          /* deep copy */
          datums_[i * max_size_ + saved_size_ + k].deep_copy(src[j], exec_ctx_->get_allocator());
          ++k;
        }
      }
    }
    saved_size_ += cnt;
  }
  return ret;
}

int ObVectorBuffer::load(ObEvalCtx &eval_ctx, ObBatchRows &brs_, int64_t batch_size) 
{
  int ret = OB_SUCCESS;
  if (NULL == exprs_) {
    // empty expr_: do nothing
  } else if (NULL == datums_) {
    ret = OB_NOT_INIT;
  } else {
    int64_t size = (saved_size_ < batch_size) ? saved_size_ : batch_size;
    for (int64_t i = 0; i < exprs_->count(); i++) {
      ObExpr *e = exprs_->at(i);
      /* shallow copy */
      MEMCPY(e->locate_batch_datums(eval_ctx), datums_ + i * max_size_, sizeof(ObDatum) * size); 
    }
    move(size);
    brs_.size_ = size;
    brs_.skip_->reset(size);
  }
  return ret;
}

int ObVectorBuffer::move(int64_t size) 
{
  int ret = OB_SUCCESS;
  if (NULL == exprs_) {
    // empty expr_: do nothing
  } else if (NULL == datums_) {
    ret = OB_NOT_INIT;
  } else if (saved_size_ > 0) {
    for (int64_t i = 0; i < exprs_->count(); i++) {
      /* shallow copy */
      MEMMOVE(datums_ + i * max_size_, datums_ + i * max_size_ + size, sizeof(ObDatum) * (saved_size_ - size)); 
    }
    saved_size_ -= size;
  }
  return ret;
}
/* --------------------------------------- end of buffer --------------------------------------*/

} // end namespace sql
} // end namespace oceanbase
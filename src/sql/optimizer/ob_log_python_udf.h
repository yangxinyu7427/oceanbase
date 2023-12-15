#ifndef _OB_LOG_PYTHON_UDF_H_
#define _OB_LOG_PYTHON_UDF_H_
#include "sql/optimizer/ob_logical_operator.h"
#include "sql/optimizer/ob_log_subplan_scan.h"

namespace oceanbase
{
namespace sql
{
class ObLogPythonUDF : public ObLogSubPlanScan
{
public:
  ObLogPythonUDF(ObLogPlan &plan)
      : ObLogSubPlanScan(plan)
  {}

  ~ObLogPythonUDF() {};
  
private:
  //common::ObSEArray<ObRawExpr*, 4, common::ModulePageAllocator, true> udf_exprs_;
  DISALLOW_COPY_AND_ASSIGN(ObLogPythonUDF);
};

} // end of namespace sql
} // end of namespace oceanbase


#endif /* OB_LOG_SUBQUERY_SCAN_H_ */

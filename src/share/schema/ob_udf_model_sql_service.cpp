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
#define USING_LOG_PREFIX SHARE_SCHEMA

#include <Python.h>

#include <iostream>
#include <fstream>
#include "ob_udf_model_sql_service.h"
#include "lib/oblog/ob_log.h"
#include "lib/oblog/ob_log_module.h"
#include "lib/string/ob_sql_string.h"
#include "lib/mysqlclient/ob_mysql_proxy.h"
#include "share/ob_dml_sql_splicer.h"
#include "share/schema/ob_schema_struct.h"
#include "share/inner_table/ob_inner_table_schema_constants.h"
namespace oceanbase
{
using namespace common;
namespace share
{
namespace schema
{
int ObUdfModelSqlService::insert_udf_model(const ObUdfModel &model_info,
                                           common::ObISQLClient *sql_client,
                                           const common::ObString *ddl_stmt_str)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(sql_client)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("sql_client is NULL, ", K(ret));
  } else if (!model_info.is_valid()) {
    ret = OB_INVALID_ARGUMENT;
    SHARE_SCHEMA_LOG(WARN, "model_info is invalid", K(model_info.get_model_name_str()), K(ret));
  } else {
    if (OB_FAIL(add_udf_model(*sql_client, model_info))) {
      LOG_WARN("failed to add model", K(ret));
    } 
    // else {
    //   ObSchemaOperation opt;
    //   opt.tenant_id_ = udf_info.get_tenant_id();
    //   opt.op_type_ = OB_DDL_CREATE_MODEL;
    //   opt.schema_version_ = udf_info.get_schema_version();
    //   opt.udf_name_ = udf_info.get_udf_name_str();
    // //   this is a trick. just like outline, synonym
    // //   use table_id_ to store there own id, we use table_id_ to store udf_id and
    // //   table name to store udf name. the reason is table_id_ and table_name_ will
    // //   be write to inner table which named all_ddl_operation, but udf_name_ will not.
    //   opt.table_id_ = udf_info.get_udf_id();
    //   opt.table_name_ = udf_info.get_udf_name_str();
    //   opt.ddl_stmt_str_ = (NULL != ddl_stmt_str) ? *ddl_stmt_str : ObString();
    //   if (OB_FAIL(log_operation(opt, *sql_client))) {
    //     LOG_WARN("Failed to log operation", K(ret));
    //   }
    // }
  }
  return ret;
}
int ObUdfModelSqlService::add_udf_model(common::ObISQLClient &sql_client,
                                        const ObUdfModel &model_info)
{
  int ret = OB_SUCCESS;
  LOG_WARN("get into ObUdfModelSqlService::add_udf_model", K(ret));
  UNUSED(sql_client);
  UNUSED(model_info);
  ObSqlString sql;
  ObSqlString values;
  //系统表
  std::string table_name = "__all_udf_model";
  ObArenaAllocator allocator(ObModIds::OB_SCHEMA);
  const uint64_t tenant_id = model_info.get_tenant_id();
  const uint64_t exec_tenant_id = ObSchemaUtils::get_exec_tenant_id(tenant_id);
  if (OB_FAIL(sql.assign_fmt("INSERT INTO %s (", table_name.c_str()))) {
      STORAGE_LOG(WARN, "append table name failed, ", K(ret));
  } else {
      LOG_WARN("=========beginning to append sql======== ", K(ret));
      SQL_COL_APPEND_VALUE(sql, values, ObSchemaUtils::get_extract_tenant_id(
                           exec_tenant_id, model_info.get_tenant_id()), "tenant_id", "%lu");
      SQL_COL_APPEND_VALUE(sql, values, ObSchemaUtils::get_extract_schema_id(
                           exec_tenant_id, model_info.get_model_id()), "model_id", "%lu");
      SQL_COL_APPEND_ESCAPE_STR_VALUE(sql, values, model_info.get_model_name(),
                                      model_info.get_model_name_str().length(), "model_name");
      SQL_COL_APPEND_VALUE(sql, values, model_info.get_framework(), "framework", "%d");
      SQL_COL_APPEND_VALUE(sql, values, model_info.get_model_type(), "model_type", "%d");
      SQL_COL_APPEND_ESCAPE_STR_VALUE(sql, values, model_info.get_model_path(),
                                      model_info.get_model_path_str().length(), "model_path");
      SQL_COL_APPEND_VALUE(sql, values, model_info.get_arg_num(), "arg_num", "%d");
      SQL_COL_APPEND_ESCAPE_STR_VALUE(sql, values, model_info.get_arg_names(),
                                      model_info.get_arg_names_str().length(), "arg_names");
      SQL_COL_APPEND_ESCAPE_STR_VALUE(sql, values, model_info.get_arg_types(),
                                      model_info.get_arg_types_str().length(), "arg_types");
      SQL_COL_APPEND_VALUE(sql, values, model_info.get_ret(), "ret", "%d");
      SQL_COL_APPEND_VALUE(sql, values, model_info.get_schema_version(), "schema_version", "%ld");
      
      if (OB_SUCC(ret)) {
        int64_t affected_rows = 0;
        if (OB_FAIL(sql.append_fmt(") VALUES (%.*s)",
                                   static_cast<int32_t>(values.length()),
                                   values.ptr()))) {
          LOG_WARN("append sql failed, ", K(ret));
        } else if (OB_FAIL(sql_client.write(exec_tenant_id, sql.ptr(), affected_rows))) {
          LOG_WARN("fail to execute sql", K(sql), K(ret));
        } else {
          if (!is_single_row(affected_rows)) {
            ret = OB_ERR_UNEXPECTED;
            LOG_WARN("unexpected value", K(affected_rows), K(sql), K(ret));
          }
        }
      }
    }
    values.reset();
    return ret;
}

int ObUdfModelSqlService::alter_udf_model(const obrpc::ObAlterUdfModelArg &alter_udf_model_arg,
                                          bool &exist,
                                          common::ObISQLClient *sql_client,
                                          const common::ObString *ddl_stmt_str)
{
  int ret = OB_SUCCESS;
  LOG_WARN("get into ObUdfModelSqlService::alter_udf_model", K(ret));
  int64_t affected_rows = 0;
  const uint64_t exec_tenant_id = alter_udf_model_arg.tenant_id_;
  common::ObString model_name = alter_udf_model_arg.model_name_;
  bool is_distillation = alter_udf_model_arg.is_distillation_;
  bool is_binarization = alter_udf_model_arg.is_binarization_;
  common::ObString model_path_before = alter_udf_model_arg.model_path_before_;
  common::ObString model_path_after = alter_udf_model_arg.model_path_after_;
  ObSqlString sql;
  if (OB_ISNULL(sql_client)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("invalid sql client is NULL", K(ret));
  } else {
    if (exist == true) {       //模型元信息存在
    //系统表
      std::string table_name = "__all_udf_model";
      if (FAILEDx(sql.assign_fmt("UPDATE %s SET model_path='%s' WHERE model_name='%s'",
                                  table_name.c_str(),
                                  model_path_after.ptr(),
                                  model_name.ptr()))) {
        LOG_WARN("append_fmt failed", K(ret));
      } else if (OB_FAIL(sql_client->write(exec_tenant_id, sql.ptr(), affected_rows))) {
        LOG_WARN("fail to execute sql", K(exec_tenant_id), K(sql), K(ret));
      } else if (1 != affected_rows) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("no row deleted", K(sql), K(affected_rows), K(ret));
      } else {/*do nothing*/}
    }
    if (is_distillation) {
    if (OB_FAIL(model_distillation(model_path_before, model_path_after))) {
        LOG_WARN("fail to execute model distillation", K(sql), K(model_path_before), K(ret));
      }
    } else if (is_binarization) {
      if (OB_FAIL(model_binarization(model_path_before, model_path_after))) {
        LOG_WARN("fail to execute model inarization", K(sql), K(model_path_before), K(ret));
      }
    }
    // // log operation
    // if (OB_SUCC(ret)) {
    //   ObSchemaOperation opt;
    //   opt.tenant_id_ = tenant_id;
    //   opt.op_type_ = OB_DDL_DROP_UDF;
    //   opt.schema_version_ = new_schema_version;
    //   opt.udf_name_ = name;
    //   //this is a trick. just like outline, synonym
    //   //use table_id_ to store there own id, we use table name to store
    //   //udf name.
    //   opt.table_name_ = name;
    //   opt.ddl_stmt_str_ = (NULL != ddl_stmt_str) ? *ddl_stmt_str : ObString();
    //   if (OB_FAIL(log_operation(opt, *sql_client))) {
    //     LOG_WARN("Failed to log operation", K(ret));
    //   }
    // }
  }
  return ret;
}
int ObUdfModelSqlService::delete_udf_model(const uint64_t tenant_id,            
                                           const common::ObString &name,
                                           const int64_t new_schema_version,
                                           common::ObISQLClient *sql_client,
                                           const common::ObString *ddl_stmt_str)
{
  int ret = OB_SUCCESS;
  int64_t affected_rows = 0;
  ObSqlString sql;
  const int64_t IS_DELETED = 1;
  const uint64_t exec_tenant_id = ObSchemaUtils::get_exec_tenant_id(tenant_id);
  if (OB_ISNULL(sql_client)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("invalid sql client is NULL", K(ret));
  } else {
    // // insert into __all_udf_history
    // if (FAILEDx(sql.assign_fmt(
    //                "INSERT INTO %s(tenant_id, name, schema_version, is_deleted)"
    //                " VALUES(%lu,'%s',%ld,%ld)",
    //                OB_ALL_FUNC_HISTORY_TNAME,
    //                ObSchemaUtils::get_extract_tenant_id(exec_tenant_id, tenant_id),
    //                name.ptr(),
    //                new_schema_version, IS_DELETED))) {
    //   LOG_WARN("assign insert into all udf history fail", K(tenant_id), K(ret));
    // } else if (OB_FAIL(sql_client->write(exec_tenant_id, sql.ptr(), affected_rows))) {
    //   LOG_WARN("execute sql fail", K(sql), K(ret));
    // } else if (1 != affected_rows) {
    //   ret = OB_ERR_UNEXPECTED;
    //   LOG_WARN("no row has inserted", K(ret));
    // } else {/*do nothing*/}
    // delete from __all_func
    //系统表
    std::string table_name = "__all_udf_model";
    if (FAILEDx(sql.assign_fmt("DELETE FROM %s WHERE tenant_id = %ld AND model_name='%s'",
                               table_name.c_str(),
                               ObSchemaUtils::get_extract_tenant_id(exec_tenant_id, tenant_id),
                               name.ptr()))) {
      LOG_WARN("append_fmt failed", K(ret));
    } else if (OB_FAIL(sql_client->write(exec_tenant_id, sql.ptr(), affected_rows))) {
      LOG_WARN("fail to execute sql", K(tenant_id), K(sql), K(ret));
    } else if (1 != affected_rows) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("no row deleted", K(sql), K(affected_rows), K(ret));
    } else {/*do nothing*/}
    // // log operation
    // if (OB_SUCC(ret)) {
    //   ObSchemaOperation opt;
    //   opt.tenant_id_ = tenant_id;
    //   opt.op_type_ = OB_DDL_DROP_UDF;
    //   opt.schema_version_ = new_schema_version;
    //   opt.udf_name_ = name;
    //   //this is a trick. just like outline, synonym
    //   //use table_id_ to store there own id, we use table name to store
    //   //udf name.
    //   opt.table_name_ = name;
    //   opt.ddl_stmt_str_ = (NULL != ddl_stmt_str) ? *ddl_stmt_str : ObString();
    //   if (OB_FAIL(log_operation(opt, *sql_client))) {
    //     LOG_WARN("Failed to log operation", K(ret));
    //   }
    // }
  }
  return ret;
}
int ObUdfModelSqlService::drop_udf_model(const ObUdfModel &model_info,
                                         const int64_t new_schema_version,
                                         common::ObISQLClient *sql_client,
                                         const common::ObString *ddl_stmt_str)
{
  int ret = OB_SUCCESS;
  ObSqlString sql;
  if (OB_ISNULL(sql_client)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("invalid sql client is NULL", K(ret));
  } else if (!model_info.is_valid()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("invalid udf info in drop udf model", K(model_info.get_model_name_str()), K(ret));
  } else if (OB_FAIL(delete_udf_model(model_info.get_tenant_id(), model_info.get_model_name(),
                                      new_schema_version, sql_client, ddl_stmt_str))) {
    LOG_WARN("failed to delete udf model", K(model_info.get_model_name_str()), K(ret));
  } else {/*do nothing*/}
  return ret;
}

int ObUdfModelSqlService::model_distillation(common::ObString &model_path_before, common::ObString &model_path_after){
  int ret = OB_SUCCESS;
  LOG_WARN("get into ObUdfModelSqlService::model_distillation", K(ret));
  // 加载教师模型
  std::string teacher_model_path = model_path_before.ptr(); // 输入路径
  // 获取学生模型保存路径
  std::string student_model_path = model_path_after.ptr();

  std::string python_code;
  python_code +=std::string("\nimport torch") +
                std::string("\nimport torch.nn as nn") +
                std::string("\nimport torch.optim as optim") +
                std::string("\nfrom torch.utils.data import DataLoader, TensorDataset") +
                std::string("\nimport pandas as pd") +
                std::string("\nimport joblib") +
                std::string("\nfrom sklearn.model_selection import train_test_split") +
                std::string("\nfrom sklearn.preprocessing import MinMaxScaler, Normalizer") +
                std::string("\nfrom sklearn.compose import ColumnTransformer") +
                std::string("\nclass ComplexMLPModel(nn.Module):") +
                std::string("\n  def __init__(self, input_size, hidden_size, output_size):") +
                std::string("\n    super(ComplexMLPModel, self).__init__()") +
                std::string("\n    self.layers = nn.Sequential(") +
                std::string("\n      nn.Linear(input_size, hidden_size),") +
                std::string("\n      nn.ReLU(),") +
                std::string("\n      nn.BatchNorm1d(hidden_size),") +
                std::string("\n      nn.Dropout(0.3),") +
                std::string("\n      nn.Linear(hidden_size, hidden_size * 2),") +
                std::string("\n      nn.ReLU(),") +
                std::string("\n      nn.BatchNorm1d(hidden_size * 2),") +
                std::string("\n      nn.Dropout(0.3),") +
                std::string("\n      nn.Linear(hidden_size * 2, hidden_size * 4),") +
                std::string("\n      nn.ReLU(),") +
                std::string("\n      nn.BatchNorm1d(hidden_size * 4),") +
                std::string("\n      nn.Dropout(0.4),") +
                std::string("\n      nn.Linear(hidden_size * 4, hidden_size * 8),") +
                std::string("\n      nn.ReLU(),") +
                std::string("\n      nn.BatchNorm1d(hidden_size * 8),") +
                std::string("\n      nn.Dropout(0.4),") +
                std::string("\n      nn.Linear(hidden_size * 8, hidden_size * 4),") +
                std::string("\n      nn.ReLU(),") +
                std::string("\n      nn.BatchNorm1d(hidden_size * 4),") +
                std::string("\n      nn.Linear(hidden_size * 4, hidden_size * 2),") +
                std::string("\n      nn.ReLU(),") +
                std::string("\n      nn.BatchNorm1d(hidden_size * 2),") +
                std::string("\n      nn.Linear(hidden_size * 2, hidden_size),") +
                std::string("\n      nn.ReLU(),") +
                std::string("\n      nn.BatchNorm1d(hidden_size),") +
                std::string("\n      nn.Linear(hidden_size, output_size)") +
                std::string("\n    )") +
                std::string("\n  def forward(self, x):") +
                std::string("\n    return self.layers(x)") +
                std::string("\nclass DistilledMLPModel(nn.Module):") +
                std::string("\n  def __init__(self, input_size, hidden_size, output_size):") +
                std::string("\n    super(DistilledMLPModel, self).__init__()") +
                std::string("\n    self.layers = nn.Sequential(") +
                std::string("\n      nn.Linear(input_size, hidden_size),") +
                std::string("\n      nn.ReLU(),") +
                std::string("\n      nn.Linear(hidden_size, output_size)") +
                std::string("\n    )") +
                std::string("\n  def forward(self, x):") +
                std::string("\n    return self.layers(x)") +
                std::string("\ndef distillation_loss(y_student, y_teacher, temperature=2.0):") +
                std::string("\n  student_prob = torch.sigmoid(y_student)") +
                std::string("\n  teacher_prob = torch.sigmoid(y_teacher)") +
                std::string("\n  return nn.BCELoss()(student_prob, teacher_prob)") +
                std::string("\ndata = pd.read_csv(\"/root/JS_test/dataset/Neo/neo.csv\").dropna()") +
                std::string("\nnumerical = ['est_diameter_min', 'est_diameter_max', 'relative_velocity', 'miss_distance', 'absolute_magnitude']") +
                std::string("\nX = data[numerical]") +
                std::string("\ny = data['hazardous'].values") +
                std::string("\npreprocessor = ColumnTransformer([") +
                std::string("\n  (\"scaler1\", MinMaxScaler(), numerical[:3]),") +
                std::string("\n  (\"scaler2\", Normalizer(), numerical[3:])") +
                std::string("\n])") +
                std::string("\nX_processed = preprocessor.fit_transform(X)") +
                std::string("\njoblib.dump(preprocessor, \"/root/JS_test/models/neo/neo_preprocessor.pkl\")") +
                std::string("\nX_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)") +
                std::string("\nX_train_tensor = torch.tensor(X_train, dtype=torch.float32)") +
                std::string("\nX_test_tensor = torch.tensor(X_test, dtype=torch.float32)") +
                std::string("\ny_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)") +
                std::string("\ny_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)") +
                std::string("\ntrain_dataset = TensorDataset(X_train_tensor, y_train_tensor)") +
                std::string("\ntrain_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)") +
                std::string("\ninput_size = 5") +
                std::string("\nhidden_size = 512") +
                std::string("\noutput_size = 1") +
                std::string("\nteacher_model = torch.load(r\"") + teacher_model_path + std::string("\", map_location='cpu')") +
                std::string("\nteacher_model.eval()") +
                std::string("\nstudent_model = DistilledMLPModel(input_size, hidden_size, output_size)") +
                std::string("\noptimizer = optim.Adam(student_model.parameters(), lr=0.01)") +
                std::string("\nscheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)") +
                std::string("\nhard_loss_fn = nn.BCELoss()") +
                std::string("\nalpha = 0.5") +
                std::string("\ntemperature = 4.0") +
                std::string("\nnum_epochs = 10") +
                std::string("\nfor epoch in range(num_epochs):") +
                std::string("\n  student_model.train()") +
                std::string("\n  for X_batch, y_batch in train_loader:") +
                std::string("\n    optimizer.zero_grad()") +
                std::string("\n    with torch.no_grad():") +
                std::string("\n      teacher_output = teacher_model(X_batch)") +
                std::string("\n    student_output = student_model(X_batch)") +
                std::string("\n    hard_loss = hard_loss_fn(torch.sigmoid(student_output), y_batch)") +
                std::string("\n    soft_loss = distillation_loss(student_output, teacher_output, temperature)") +
                std::string("\n    loss = alpha * hard_loss + (1 - alpha) * soft_loss") +
                std::string("\n    loss.backward()") +
                std::string("\n    optimizer.step()") +
                std::string("\n  scheduler.step()") +
                std::string("\ntorch.save(student_model, r\"") + student_model_path + std::string("\")");


  // Python 执行准备
  PyObject *pModule = nullptr;
  PyObject *pGlobals = nullptr;
  PyObject *py_result = nullptr;

  //Acquire GIL
  bool nStatus = PyGILState_Check();
  PyGILState_STATE gstate;
  if(!nStatus) {
    gstate = PyGILState_Ensure();
    nStatus = true;
  }

  // 获取 __main__ 模块并执行
  if ((pModule = PyImport_AddModule("__main__")) == nullptr) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("failed to import __main__ module", K(ret));
  } else if ((pGlobals = PyModule_GetDict(pModule)) == nullptr) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("failed to get __main__ dict", K(ret));
  } else if ((py_result = PyRun_StringFlags(python_code.c_str(), Py_file_input, pGlobals, pGlobals, NULL)) == nullptr) {
    // 捕获 Python 错误
    process_python_exception();
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("Failed to run Python model distillation script", K(ret));
  } else {
    LOG_WARN("Python model distillation executed successfully", K(ret));
  }
  return ret;
}

int ObUdfModelSqlService::model_binarization(common::ObString &model_path_before, common::ObString &model_path_after){
  int ret = OB_SUCCESS;
  // 学生模型保存路径
  std::string student_model_path = model_path_before.ptr(); // 输入路径
  // 获取二值化后的保存路径
  std::string binary_model_path = model_path_after.ptr();
  std::string python_code = R"(
import torch
import torch.nn as nn

def binarize(tensor, mode='deterministic'):
    if mode == 'deterministic':
        return tensor.sign()
    elif mode == 'stochastic':
        return (tensor > torch.rand(tensor.size()).to(tensor.device)).float() * 2 - 1

class BinaryLinear(nn.Linear):
    def forward(self, input):
        binary_weight = binarize(self.weight)
        input = binarize(input)
        return nn.functional.linear(input, binary_weight, self.bias)

class BinaryMLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BinaryMLPModel, self).__init__()
        self.layers = nn.Sequential(
            BinaryLinear(input_size, hidden_size),
            nn.ReLU(),
            BinaryLinear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

class DistilledMLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DistilledMLPModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

input_size = 5
hidden_size = 512
output_size = 1

teacher_model = torch.load(r')" + student_model_path + R"(')
binary_model = BinaryMLPModel(input_size, hidden_size, output_size)

with torch.no_grad():
    for bin_layer, teacher_layer in zip(binary_model.layers, teacher_model.layers):
        if isinstance(bin_layer, BinaryLinear) and isinstance(teacher_layer, nn.Linear):
            bin_layer.weight.data.copy_(teacher_layer.weight.data)
            if teacher_layer.bias is not None:
                bin_layer.bias.data.copy_(teacher_layer.bias.data)

torch.save(binary_model, r')" + binary_model_path + R"(')
print("Binary model saved successfully.")
)";

  // Python 执行准备
  PyObject *pModule = nullptr;
  PyObject *pGlobals = nullptr;
  PyObject *py_result = nullptr;

  //Acquire GIL
  bool nStatus = PyGILState_Check();
  PyGILState_STATE gstate;
  if(!nStatus) {
    gstate = PyGILState_Ensure();
    nStatus = true;
  }

  // 获取 __main__ 模块并执行
  if ((pModule = PyImport_AddModule("__main__")) == nullptr) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("failed to import __main__ module", K(ret));
  } else if ((pGlobals = PyModule_GetDict(pModule)) == nullptr) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("failed to get __main__ dict", K(ret));
  } else if ((py_result = PyRun_StringFlags(python_code.c_str(), Py_file_input, pGlobals, pGlobals, NULL)) == nullptr) {
    // 捕获 Python 错误
    process_python_exception();
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("Failed to run Python model distillation script", K(ret));
  } else {
    LOG_WARN("Python model distillation executed successfully", K(ret));
  }
  return ret;
}

} //end of schema
} //end of share
} //end of oceanbase
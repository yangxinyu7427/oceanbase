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
#ifndef OB_MODEL_SQL_SERVICE_H_
#define OB_MODEL_SQL_SERVICE_H_
#include "ob_ddl_sql_service.h"
namespace oceanbase
{
namespace common
{
class ObString;
class ObISQLClient;
}
namespace share
{
namespace schema
{
class ObUdfModel;
class ObUdfModelSqlService : public ObDDLSqlService
{
public:
  ObUdfModelSqlService(ObSchemaService &schema_service)
    : ObDDLSqlService(schema_service) {}
  virtual ~ObUdfModelSqlService() {}
  virtual int insert_udf_model(const ObUdfModel &model_info,
                               common::ObISQLClient *sql_client,
                               const common::ObString *ddl_stmt_str = NULL);
  virtual int delete_udf_model(const uint64_t tenant_id,
                               const common::ObString &name,
                               const int64_t new_schema_version,
                               common::ObISQLClient *sql_client,
                               const common::ObString *ddl_stmt_str = NULL);
  virtual int drop_udf_model(const ObUdfModel &model_info,
                             const int64_t new_schema_version,
                             common::ObISQLClient *sql_client,
                             const common::ObString *ddl_stmt_str = NULL);
private:
  int add_udf_model(common::ObISQLClient &sql_client, 
                     const ObUdfModel &model_info);
private:
  DISALLOW_COPY_AND_ASSIGN(ObUdfModelSqlService);
};
} //end of namespace schema
} //end of namespace share
} //end of namespace oceanbase
#endif
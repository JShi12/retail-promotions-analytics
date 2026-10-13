# Creating the scheduled job

The job runs the whole project on serverless compute. Its intended definition is
[`jobs/retail_pipeline.yml`](../jobs/retail_pipeline.yml); these steps create it in the UI.

1. **Jobs & Pipelines → Create → Job.** Name it `retail-promotions-pipeline`.
2. **Job parameters** (job details panel → *Edit parameters*):
   - `catalog` = `retail`
   - `run_id` = `{{job.run_id}}` (one id for the whole run; every notebook reads it from its widget)
3. **Tasks.** For each row, *Add task → Notebook*, choose the notebook from your Git folder, compute
   *Serverless*, and set *Depends on*:

   | Task name | Notebook | Depends on |
   |---|---|---|
   | `setup` | `notebooks/00_setup` | — |
   | `bronze` | `notebooks/01_bronze` | setup |
   | `silver` | `notebooks/02_silver` | bronze |
   | `gold` | `notebooks/03_gold` | silver |
   | `dq_report` | `notebooks/04_dq_report` | gold |
   | `promo_effectiveness` | `notebooks/10_promo_effectiveness` | gold |
   | `customer_segments` | `notebooks/20_customer_segments` | gold |
   | `coupon_response` | `notebooks/30_coupon_response` | gold |
   | `dashboard_views` | `notebooks/40_dashboard_views` | promo_effectiveness, customer_segments, coupon_response |

4. **Schedule** (job details → *Add trigger → Scheduled*): weekly, Monday 06:00, Europe/London.
5. **Check against the repo:** ⋮ next to *Run now* → *Switch to code version (YAML)* and compare with
   `jobs/retail_pipeline.yml` (task names, dependencies, parameters, schedule).
6. **Run once:** *Run now*, at a quiet time. The first full run took 46 minutes.
7. **Pause** the schedule after the run succeeds (job details → schedule → *Pause*). The data is
   static, so the schedule demonstrates the setup; pausing protects the Free Edition daily quota.

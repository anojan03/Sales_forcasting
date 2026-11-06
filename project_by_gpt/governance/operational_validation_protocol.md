# Prospective operational validation plan

Status: ready-to-adapt study protocol, not an executed business-impact evaluation. Revenue peak precision/recall and over/underforecast totals have already been calculated from recorded sales; actual staffing savings, inventory improvements and profit changes have not.

## Define the decision before collecting outcomes

Choose staffing or a small set of inventory items as the primary decision. Freeze the forecast model, origin, thresholds, comparator policy and analysis before the pilot. Use future trading periods that were not inspected during model development. Record closures, extraction failures, promotions, price changes and policy overrides rather than treating all changes as model effects.

## Staffing pilot

Collect timestamp, receipt count, scheduled staff-hours, actual staff-hours, service-time or queue measurements, wages and opening status. Avoid employee names and customer identifiers in the research table. Specify a capacity rule using measured transaction/service workload rather than assuming each pound of revenue requires equal labour. Compare the forecast-assisted rule with the store's existing practice using a practicable crossover or staggered design agreed with the store. Measure service outcomes and labour costs together; lower labour cost with materially worse queues is not automatically success.

## Inventory pilot

For selected SKUs collect date, pseudonymous product code, opening stock, replenishments, recorded units sold, closing stock, verified stockout periods, selling price, acquisition cost, spoilage, lead time and order decisions. Revenue alone cannot recover lost demand while stock is unavailable. Predefine service-level and holding/shortage cost measures and account for stockout censoring. Department net-revenue models cannot be directly interpreted as SKU reorder quantities without a separate quantity forecast and inventory policy.

## Evaluation

Record both planned and actually implemented decisions. Report missingness and adherence. Compare matched trading periods and account for temporal dependence and changes in promotions, holidays and demand. Treat financial savings as model-associated unless the design supports stronger attribution. Include implementation cost and adverse outcomes. Use a justified sample-size or precision plan based on pilot variability; do not invent a number of weeks that guarantees significance.

CSV headers in governance/templates define candidate fields. They are intentionally empty. Neither file is evidence that a pilot has occurred. After real observations are available, agree the final policy and analysis with the supervisor before calculating business-impact claims.

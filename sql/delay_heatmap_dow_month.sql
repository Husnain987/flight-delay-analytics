-- delay_heatmap_dow_month.csv
-- Day-of-week x month grid (avg arrival delay) for heatmap
SELECT
  day_of_week,
  month,
  COUNT(*)                                                    AS total_flights,
  ROUND(SAFE_DIVIDE(SUM(CASE WHEN is_delayed THEN 1 ELSE 0 END), COUNT(*)) * 100, 2) AS delay_rate_pct,
  ROUND(AVG(arr_delay), 2)                                    AS avg_arr_delay_min
FROM `flights.flights_2024`
GROUP BY day_of_week, month
ORDER BY month, day_of_week;
-- delay_by_dow.csv
-- Delay rate by day of week
SELECT
  day_of_week,
  COUNT(*)                                                    AS total_flights,
  SUM(CASE WHEN is_delayed THEN 1 ELSE 0 END)                 AS delayed_flights,
  ROUND(SAFE_DIVIDE(SUM(CASE WHEN is_delayed THEN 1 ELSE 0 END), COUNT(*)) * 100, 2) AS delay_rate_pct,
  ROUND(AVG(dep_delay), 2)                                    AS avg_dep_delay_min
FROM `flights.flights_2024`
GROUP BY day_of_week
ORDER BY
  CASE day_of_week
    WHEN 'Monday'    THEN 1
    WHEN 'Tuesday'   THEN 2
    WHEN 'Wednesday' THEN 3
    WHEN 'Thursday'  THEN 4
    WHEN 'Friday'    THEN 5
    WHEN 'Saturday'  THEN 6
    WHEN 'Sunday'    THEN 7
  END;

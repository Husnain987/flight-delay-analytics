-- worst_routes.csv
-- Worst origin->destination routes by delay rate
SELECT
  route,
  COUNT(*)                                                    AS total_flights,
  SUM(CASE WHEN is_delayed THEN 1 ELSE 0 END)                 AS delayed_flights,
  ROUND(SAFE_DIVIDE(SUM(CASE WHEN is_delayed THEN 1 ELSE 0 END), COUNT(*)) * 100, 2) AS delay_rate_pct,
  ROUND(AVG(dep_delay), 2)                                    AS avg_dep_delay_min
FROM `flights.flights_2024`
GROUP BY route
HAVING total_flights > 1000
ORDER BY delay_rate_pct DESC;
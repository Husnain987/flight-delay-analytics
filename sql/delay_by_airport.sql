-- delay_by_airport.csv
-- Delay rate by origin airport (busiest airports only)
SELECT
  origin,
  COUNT(*)                                                    AS total_flights,
  SUM(CASE WHEN is_delayed THEN 1 ELSE 0 END)                 AS delayed_flights,
  ROUND(SAFE_DIVIDE(SUM(CASE WHEN is_delayed THEN 1 ELSE 0 END), COUNT(*)) * 100, 2) AS delay_rate_pct,
  ROUND(AVG(dep_delay), 2)                                    AS avg_dep_delay_min
FROM `flights.flights_2024`
GROUP BY origin
HAVING total_flights > 5000
ORDER BY total_flights DESC
LIMIT 30;
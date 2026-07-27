SELECT internet_service, COUNT(*) AS count
        FROM services JOIN customers ON customers.service_id = services.id
        GROUP BY internet_service
        ORDER BY count DESC;

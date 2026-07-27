SELECT payment_method, COUNT(*) AS count
        FROM contracts JOIN customers ON customers.contract_id = contracts.id
        GROUP BY payment_method
        ORDER BY count DESC;

SELECT contract_type, COUNT(*) AS count
        FROM contracts JOIN customers ON customers.contract_id = contracts.id
        GROUP BY contract_type
        ORDER BY count DESC;

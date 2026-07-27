SELECT customer_id, total_charges
        FROM customers JOIN billings ON customers.billing_id = billings.id
        ORDER BY total_charges DESC
        LIMIT 10;

SELECT 
	i.name,i.object_id,SUM(s.used_page_count)*8 
FROM 
	sys.dm_db_partition_stats AS s 
    INNER JOIN 
    sys.indexes AS i 
    ON s.object_id = i.object_id 
    AND 
    s.index_id = i.index_id 
GROUP BY 
	i.name,i.object_id
ORDER BY 
	i.name 
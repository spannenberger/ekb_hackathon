CREATE DATABASE ekb_service
    DEFAULT CHARACTER SET = 'utf8mb4';

USE ekb_service
CREATE TABLE service_results(  
    id int NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT 'Primary Key',
    update_time DATETIME COMMENT 'Update Time',
    service_result TEXT(5000) COMMENT 'service result',
    conclusion TEXT(5000) COMMENT 'conclusion'
) DEFAULT CHARSET UTF8 COMMENT 'newTable';

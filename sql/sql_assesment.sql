#create database new;
use new;
/*drop table if exists customer;
drop table if exists city;
drop table if exists country;*/

/*CREATE TABLE country(
  country_id INT NOT NULL,
  country_name VARCHAR(50),
  country_name_eng VARCHAR(50),
  country_code VARCHAR(20),
  PRIMARY KEY (country_id)
);*/

/*CREATE TABLE city(
	city_id INT PRIMARY KEY,
    city_name VARCHAR(50),
    lat FLOAT8,
    city_long float8,
    country_id INT,
    FOREIGN KEY (country_id) REFERENCES country(country_id)
);*/

/*CREATE TABLE customer(
	customer_id INT,
    customer_name VARCHAR(50),
    city_id INT,
    customer_address VARCHAR(50),
    next_call_date DATE,
    ts_inserted DATETIME,
    FOREIGN KEY (city_id) REFERENCES city (city_id)
);*/


/*insert into country values(1,'Deutschland','Gemany','DEU');
insert into country values(2,'Srbija','Serbia','SRB');
insert into country values(3,'Hrvatska','Croatia','HRV');
insert into country values(4,'United States of America','United States of America','USA');
insert into country values(5,'Polska','Poland','POL');
insert into country values(6,'Espana','Spain','ESP');
insert into country values(7,'Rossiya','Russia','RUS');*/

/*insert into city
values(1,'Berlin',52.520008,13.404954,1),
(2,'Belgrade',44.787197,20.457273,2),
(3,'Zagreb',45.815399,15.966568,3),
(4,'New York',40.730610,-73.935242,4),
(5,'Los Angeles',34.052235,-118.243683,4),
(6,'Warsaw',52.237049,21.017532,5);*/

/*insert into customer 
values(1,'Jewelry Store',4,'Long Street 120','2020-01-21','2020-01-09 14:01:20.000'),
(2,'Bakery',1,'Kurfurstendamm 25','2020-02-21','2020-01-09 17:52:15.000'),
(3,'Cafe',1,'TauentzienstraBe 44','2020-01-21','2020-01-10 08:02:49.000'),
(4,'Restaurant',3,'Ulica lipa 15','2020-01-21','2020-01-10 09:20:21.000');*/

SELECT * FROM country;

SELECT * FROM city;

SELECT * FROM customer;

/*Task : 1 (join multiple tables using left join)
List all Countries and customers related to these countries.
For each country displaying its name in English, the name of the city customer is located in as
well as the name of the customer.
Return even countries without related cities and customers.*/

/*SELECT 
	country.country_name,
    city.city_name,
    customer.customer_name
FROM country
LEFT JOIN city ON country.country_id = city.country_id
LEFT JOIN customer ON city.city_id = customer.city_id
order by country.country_name,city.city_name,customer.customer_name;
*/
/*Task : 2 (join multiple tables using both left and inner join)
Return the list of all countries that have pairs(exclude countries which are not referenced by any
city). For such pairs return all customers.
Return even pairs of not having a single customer*/

/*SELECT 
	country.country_name,
    city.city_name,
    customer.customer_name
FROM country
JOIN city ON country.country_id = city.country_id
LEFT JOIN customer ON city.city_id = customer.city_id
order by country.country_name,city.city_name,customer.customer_name;
*/
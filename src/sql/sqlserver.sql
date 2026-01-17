create table polygon (id int not null, geog geography, constraint PK_T PRIMARY KEY (ID));
create table polygon_target (id int not null, geog geography, constraint PK_T_target PRIMARY KEY (ID));
create table polygon_parent (id int not null, geog geography, constraint PK_T_parent PRIMARY KEY (ID));
create table polygon_child (id int not null, geog geography, constraint PK_T_child PRIMARY KEY (ID));

create table point (id int not null, geog geography, constraint PK_T PRIMARY KEY (ID))
CREATE SPATIAL INDEX polygon_grid on polygon(geog) USING GEOGRAPHY_AUTO_GRID
select count(*) from point t1, polygon t2 where t2.geog.STContains(t1.geog)=1 and t1.id=1

CREATE SPATIAL INDEX polygon_grid ON polygon(geog) USING GEOGRAPHY_GRID WITH ( BOUNDING_BOX = ( xmin=-180, ymin=-90, xmax=180, ymax=90), GRIDS = (MEDIUM, LOW, MEDIUM, HIGH ), CELLS_PER_OBJECT = 640, PAD_INDEX  = ON );  

CREATE SPATIAL INDEX grid_index on polygon(geog) USING GEOGRAPHY_AUTO_GRID

CREATE SPATIAL INDEX grid_index ON polygon_small(geog) USING GEOGRAPHY_GRID WITH ( GRIDS = (HIGH, LOW, low, HIGH ), CELLS_PER_OBJECT = 640, PAD_INDEX  = ON ); 

CREATE SPATIAL INDEX polygon_grid ON polygon(geog) USING GEOGRAPHY_GRID WITH ( BOUNDING_BOX = ( xmin=-180, ymin=-90, xmax=180, ymax=90), GRIDS = (MEDIUM, LOW, MEDIUM, HIGH ), CELLS_PER_OBJECT = 8192, PAD_INDEX  = ON );  


time sqlcmd -S localhost -U SA -P 'passwd' -d tengdb -I -Q "select count(*) from point t1, polygon_small t2 where t2.geog.STContains(t1.geog)=1;"
sqlcmd -S localhost -U SA -P 'passwd' -d tengdb -I -i polygon.small.sql

BULK INSERT polygon_parent FROM '/home/git/IDEAL/src/mssql.parent.csv' WITH (FIELDTERMINATOR = '|', ROWTERMINATOR = '\n');

select t1.id from point t1, polygon t2


CREATE SPATIAL INDEX polygon_parent_idx ON polygon_parent(geog);

CREATE SPATIAL INDEX polygon_parent_grid_idx ON polygon_parent(geog) USING GEOGRAPHY_GRID WITH ( GRIDS = (MEDIUM, LOW, MEDIUM, HIGH ), CELLS_PER_OBJECT = 8192, PAD_INDEX  = ON );  

CREATE SPATIAL INDEX polygon_child_idx ON polygon_child(geog);

CREATE SPATIAL INDEX polygon_child_grid_idx ON polygon_child(geog) USING GEOGRAPHY_GRID WITH ( GRIDS = (MEDIUM, LOW, MEDIUM, HIGH ), CELLS_PER_OBJECT = 8192, PAD_INDEX  = ON );  


select count(*) from polygon_parent p, polygon_child c where p.geog.STContains(c.geog)=1 and c.id=1;

set statistics time on 

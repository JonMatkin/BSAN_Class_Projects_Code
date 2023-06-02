-- This File creates all the tables for the factSharkAttack

USE Shark
GO

-- Dropping the Tables
IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[factSharkAttack]') AND type in (N'U'))
DROP TABLE [dbo].[factSharkAttack]
GO

IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[dimType]') AND type in (N'U'))
DROP TABLE [dbo].[dimType]
GO

IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[dimCountry]') AND type in (N'U'))
DROP TABLE [dbo].[dimCountry]
GO

IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[dimRegion]') AND type in (N'U'))
DROP TABLE [dbo].[dimRegion]
GO

IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[dimActivity]') AND type in (N'U'))
DROP TABLE [dbo].[dimActivity]
GO

IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[dimInjury]') AND type in (N'U'))
DROP TABLE [dbo].[dimInjury]
GO

IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[dimFatality]') AND type in (N'U'))
DROP TABLE [dbo].[dimFatality]
GO

IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[dimFatality]') AND type in (N'U'))
DROP TABLE [dbo].[dimFatality]
GO

IF EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[dimTime]') AND type in (N'U'))
DROP TABLE [dbo].[dimTime]
GO

IF EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[dimSpecies]') AND type in (N'U'))
DROP TABLE [dbo].[dimSpecies]
GO


-- Create the dimType Table
CREATE TABLE [dbo].[dimType](
	TypeID int IDENTITY(1,1) NOT NULL
		CONSTRAINT PK_dimType_TypeID PRIMARY KEY CLUSTERED (TypeID),
	TypeDescription varchar(20)
)

INSERT INTO dimType
	SELECT DISTINCT AttackType
	FROM SharkStagingTable
	ORDER BY AttackType ASC

--SELECT *
--FROM dimType

CREATE TABLE [dbo].[dimRegion](
	RegionID int IDENTITY(1,1)
		CONSTRAINT PK_dimRegion_RegionID PRIMARY KEY CLUSTERED (RegionID),
	RegionName varchar(20)
)

INSERT INTO dimRegion 
	SELECT DISTINCT Region
	FROM SharkStagingTable
	ORDER BY Region ASC

--SELECT *
--FROM dimRegion

CREATE TABLE [dbo].[dimCountry](
	CountryID int Identity(1,1) NOT NULL
		CONSTRAINT PK_dimCountry_CountryID PRIMARY KEY CLUSTERED (CountryID),
	CountryName varchar(35),
	RegionID int
		CONSTRAINT FK_dimRegion_dimCountry FOREIGN KEY (RegionID)
		REFERENCES dimRegion (RegionID)
)

INSERT INTO dimCountry
	SELECT DISTINCT Country,
	RegionID = 
		CASE
			WHEN Region Like '%North%' THEN 7
			WHEN Region LIKE '%Atlantic%' THEN 3
			WHEN Region LIKE '%Asia%' THEN 2
			WHEN Region LIKE '%Africa%' THEN 1
			WHEN Region LIKE '%Oceania%' THEN 8
			WHEN Region LIKE '%Australia%' THEN 4
			WHEN Region LIKE '%Carib%' THEN 5
			WHEN Region LIKE '%South%' THEN 10
			WHEN Region LIKE '%Pacific%' THEN 9
			WHEN Region LIKE '%Europe%' THEN 6
		END
	FROM SharkStagingTable

--SELECT *
--FROM dimCountry

CREATE TABLE [dbo].[dimActivity](
	ActivityID int IDENTITY(1,1)
		CONSTRAINT PK_dimActivity_ActivityID PRIMARY KEY CLUSTERED (ActivityID),
	ActivityDescription varchar(20)
)

INSERT INTO dimActivity
	SELECT DISTINCT Activity
	FROM SharkStagingTable
	ORDER BY Activity ASC

--SELECT *
--FROM dimActivity


CREATE TABLE [dbo].[dimInjury](
	InjuryID int IDENTITY(1,1)
		CONSTRAINT PK_dimInjury_InjuryID PRIMARY KEY CLUSTERED (InjuryID),
	InjuryDescription varchar(30)
)


INSERT INTO dimInjury
	SELECT DISTINCT Injury
	FROM SharkStagingTable
	ORDER BY Injury ASC

--SELECT *
--FROM dimInjury

CREATE TABLE [dbo].[dimFatality](
	FatalityID int
		CONSTRAINT PK_dimFatality_FatalityID PRIMARY KEY CLUSTERED (FatalityID),
	FatalityDescription varchar(10)
)

INSERT INTO dimFatality
	VALUES
	(1,NULL),
	(2,'Y'),
	(3,'N'),
	(4,'UNKNOWN')

--SELECT *
--FROM dimFatality

CREATE TABLE [dbo].[dimTime](
	TimeID int
		CONSTRAINT PK_dimTime_TimeID PRIMARY KEY CLUSTERED (TimeID),
	TimeDescription varchar(25)
)

INSERT INTO dimTime
	VALUES
		(1,'Early Morning'),
		(2,'Late Morning'),
		(3, 'Afternoon'),
		(4,'Evening'),
		(5,'Night')

--SELECT *
--FROM dimTime

CREATE TABLE [dbo].[dimSpecies](
	SpeciesID int IDENTITY(1,1)
		CONSTRAINT PK_dimSpecies_SpeciesID PRIMARY KEY CLUSTERED (SpeciesID),
	SpeciesName varchar(30)
)

INSERT INTO dimSpecies
	SELECT DISTINCT Species
	FROM SharkStagingTable
	ORDER BY Species ASC

--SELECT *
--FROM dimSpecies

CREATE TABLE [dbo].[factSharkAttack](
	AttackID int IDENTITY(1,1)
		CONSTRAINT PK_factSharkAttack_AttackID PRIMARY KEY CLUSTERED (AttackID),
	AttackDate date,
	TypeID int
		CONSTRAINT FK_dimType_factSharkAttack FOREIGN KEY (TypeID)
		REFERENCES dimType (TypeID),
	CountryID int
		CONSTRAINT FK_dimCountry_factSharkAttack FOREIGN KEY (CountryID)
		REFERENCES dimCountry (CountryID),
	ActivityID int
		CONSTRAINT FK_dimActivity_factSharkAttack FOREIGN KEY (ActivityID)
		REFERENCES dimActivity (ActivityID),
	InjuryID int
		CONSTRAINT FK_dimInjury_factSharkAttack FOREIGN KEY (InjuryID)
		REFERENCES dimInjury (InjuryID),
	FatalityID int
		CONSTRAINT FK_dimFatality_factSharkAttack FOREIGN KEY (FatalityID)
		REFERENCES dimFatality (FatalityID),
	SpeciesID int
		CONSTRAINT FK_dimSpecies_factSharkAttack FOREIGN KEY (SpeciesID)
		REFERENCES dimSpecies (SpeciesID),
	TimeID int
		CONSTRAINT FK_dimTime_factSharkAttack FOREIGN KEY (TimeID)
		REFERENCES dimTime (TimeID)
)

INSERT INTO factSharkAttack
	SELECT 
	AttackDate,
	TypeID = 
		CASE
			WHEN AttackType LIKE 'Provoked' THEN 1
			WHEN AttackType LIKE '%Quest%' THEN 2
			WHEN AttackType LIKE '%Disaster%' THEN 3
			WHEN AttackTYPE LIKE 'Unprovoked' THEN 4
		END,
	CountryID =
		CASE
			WHEN Country LIKE '%ZEALAND%' THEN 1
			WHEN Country LIKE 'JAMAICA' THEN 2
			WHEN Country LIKE '%PHIL%' THEN 3
			WHEN Country LIKE '%EGY%' THEN 4
			WHEN Country LIKE '%ITALY%' then 5
			WHEN Country LIKE '%REUN%' THEN 6
			WHEN Country LIKE '%ATLANTIC%' THEN 7
			WHEN Country LIKE '%MOZ%' THEN 8
			WHEN Country LIKE '%ECUADOR%' THEN 9
			WHEN Country LIKE '%JAPAN%' THEN 10
			WHEN Country LIKE '%AUSTRALIA%' THEN 11
			WHEN Country LIKE '%PACIFIC OCEAN%' THEN 12
			WHEN Country LIKE '%Papua New Guinea' THEN 13
			WHEN Country LIKE '%CHINA%' THEN 14
			WHEN Country LIKE 'CROATIA' THEN 15
			WHEN Country LIKE 'BRAZIL' THEN 16
			WHEN Country LIKE 'INDONESIA' THEN 17
			WHEN Country LIKE 'GRENEDA' THEN 18
			WHEN Country LIKE 'SOUTH AFRICA' THEN 19
			WHEN Country LIKE 'GREECE' THEN 20
			WHEN Country LIKE 'BAHAMAS' THEN 21
			WHEN Country LIKE 'USA' THEN 22
			WHEN Country LIKE 'VANUATU' THEN 23
			WHEN Country LIKE 'CHILE' THEN 24
			WHEN Country LIKE 'FIJI' THEN 25
			WHEN Country LIKE 'MEXICO' THEN 26
			WHEN Country LIKE 'SOLOMON ISLANDS' THEN 27
		END,
	ActivityID = 
		CASE
			WHEN Activity LIKE 'Bathing' THEN 1
			WHEN Activity LIKE 'Body Boarding' THEN 2
			WHEN Activity LIKE 'Body Surfing' THEN 3
			WHEN Activity LIKE 'Boogie Boarding' THEN 4
			WHEN Activity LIKE 'Diving' THEN 5
			WHEN Activity LIKE 'Fishing' THEN 6
			WHEN Activity LIKE 'Kayaking' THEN 7
			WHEN Activity LIKE 'Sea Disaster' THEN 8
			WHEN Activity LIKE 'Snorkeling' THEN 9
			WHEN Activity LIKE 'Spearfishing' THEN 10
			WHEN Activity LIKE 'Standing' THEN 11
			WHEN Activity LIKE 'Surfing' THEN 12
			WHEN Activity LIKE 'Swimming' THEN 13
			WHEN Activity LIKE 'Wading' THEN 14
		END,
	InjuryId =
		CASE
			WHEN Injury IS NULL THEN 1
			WHEN Injury LIKE 'Abrasions' THEN 2
			WHEN Injury LIKE 'Bitten' THEN 3
			WHEN Injury LIKE 'Cuts' THEN 4
			WHEN Injury LIKE '%Drown%' THEN 5
			WHEN Injury LIKE '%FATAL%' THEN 6
			WHEN Injury LIKE 'Laceration' THEN 7
			WHEN Injury LIKE '%Minor%' THEN 8
			WHEN Injury LIKE 'No details' THEN 9
			WHEN Injury LIKE 'No injury' THEN 10
			WHEN Injury LIKE 'Other%' THEN 11
			WHEN Injury LIKE 'Puncture%' THEN 12
			WHEN Injury LIKE 'Severe%' THEN 13
			WHEN Injury LIKE '%Shark%' THEN 14
		END,
	FatalityID = 
		CASE
			WHEN Fatality IS NULL THEN 1
			WHEN Fatality = 'Y' THEN 2
			WHEN Fatality = 'N' THEN 3
			ELSE 4
		END,
	SpeciesID = 
		CASE
			WHEN Species LIKE '%Blacktip%' THEN 1
			WHEN Species LIKE '%Bull%' THEN 2
			WHEN Species LIKE '%Lemon%' THEN 3
			WHEN Species LIKE '%Mako%' THEN 4
			WHEN Species LIKE '%Nurse%' THEN 5
			WHEN Species LIKE '%Other%' THEN 6
			WHEN Species LIKE '%Reef%' THEN 7
			WHEN Species LIKE '%Involvement%' THEN 8
			WHEN Species LIKE '%Spinner%' THEN 9
			WHEN Species LIKE '%Tiger%' THEN 10
			WHEN Species LIKE '%Unspecified%' THEN 11
			WHEN Species LIKE '%White%' THEN 12
			WHEN Species LIKE '%Wobbewong%' THEN 13
		END,
	TimeID = 
		CASE
			WHEN AttackTime LIKE '4:%' THEN 1
			WHEN AttackTime LIKE '5:%' THEN 1
			WHEN AttackTime LIKE '6:%' THEN 1
			WHEN AttackTime LIKE '7:%' THEN 1
			WHEN AttackTime LIKE '8:%' THEN 1
			WHEN AttackTime LIKE '%dawn%' THEN 1
			WHEN AttackTime LIKE 'Early Morning' THEN 1
			WHEN AttackTime LIKE '9:%' THEN 2
			WHEN AttackTime LIKE '10:%' THEN 2
			WHEN AttackTime LIKE '11:%' THEN 2
			WHEN AttackTime LIKE 'Morning' THEN 2
			WHEN AttackTime LIKE '12:%' THEN 3
			WHEN AttackTime LIKE '13:%' THEN 3
			WHEN AttackTime LIKE '14:%' THEN 3
			WHEN AttackTime LIKE '15:%' THEN 3
			WHEN AttackTime LIKE '16:%' THEN 3
			WHEN AttackTime LIKE '%noon%' THEN 3
			WHEN AttackTime LIKE '17:%' THEN 4
			WHEN AttackTime LIKE '18:%' THEN 4
			WHEN AttackTime LIKE '19:%' THEN 4
			WHEN AttackTime LIKE '20:%' THEN 4
			WHEN AttackTime LIKE '%evening%' THEN 4
			WHEN AttackTime LIKE '%sunset%' THEN 4
			WHEN AttackTime LIKE '21:%' THEN 5
			WHEN AttackTime LIKE '22:%' THEN 5
			WHEN AttackTime LIKE '23:%' THEN 5
			WHEN AttackTime LIKE '0:%' THEN 5
			WHEN AttackTime LIKE '1:%' THEN 5
			WHEN AttackTime LIKE '2:%' THEN 5
			WHEN AttackTime LIKE '3:%' THEN 5
			WHEN AttackTime LIKE '%dusk%' THEN 5
			WHEN AttackTime LIKE '%dark%' THEN 5
			WHEN AttackTime LIKE '%night%' THEN 5
		END
	FROM SharkStagingTable

SELECT *
FROM factSharkAttack
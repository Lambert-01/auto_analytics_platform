# NISR Rwanda Analytics Platform - Data Directory Structure

This directory contains all data files, models, and outputs for the NISR Rwanda Analytics Platform.

## Directory Structure

### 📁 `/uploads`
- **Purpose**: Temporary storage for uploaded files
- **Contents**: Raw CSV, Excel, JSON, Parquet files
- **Retention**: 30 days (configurable)
- **Security**: Virus scanning, file validation

### 📁 `/processed`
- **Purpose**: Cleaned and preprocessed datasets
- **Contents**: Validated, cleaned, and transformed data
- **Format**: Standardized Parquet files for performance
- **Metadata**: JSON schema files with column descriptions

### 📁 `/models`
- **Purpose**: Trained machine learning models and artifacts
- **Subfolders**:
  - `classification/` - Classification models
  - `regression/` - Regression models
  - `clustering/` - Clustering models
  - `time_series/` - Time series forecasting models
  - `nlp/` - Natural language processing models
- **Contents**: Model files, training logs, performance metrics

### 📁 `/reports`
- **Purpose**: Generated analytical reports
- **Contents**: PDF, HTML, Word documents
- **Organization**: By date, dataset, or department
- **Retention**: 7 years (NISR compliance)

### 📁 `/temp`
- **Purpose**: Temporary files during processing
- **Contents**: Intermediate processing files
- **Cleanup**: Automatic cleanup after 24 hours
- **Security**: Encrypted temporary files

### 📁 `/backup`
- **Purpose**: System backups and data snapshots
- **Contents**: Compressed database dumps, model backups
- **Schedule**: Daily incremental, weekly full backups
- **Retention**: 90 days

### 📁 `/census`
- **Purpose**: Rwanda census data and analysis
- **Contents**: Population data, demographic analysis
- **Years**: 2012, 2022, and historical data
- **Granularity**: National, provincial, district, sector levels

### 📁 `/economic`
- **Purpose**: Economic indicators and financial data
- **Contents**: GDP, inflation, trade statistics
- **Sources**: BNR, MINECOFIN, trading partners
- **Frequency**: Monthly, quarterly, annual data

### 📁 `/surveys`
- **Purpose**: National and household survey data
- **Contents**: DHS, LSMS, labor force surveys
- **Processing**: Statistical weighting, sampling corrections
- **Privacy**: Anonymized, PII removed

### 📁 `/geo`
- **Purpose**: Geographic and spatial data
- **Contents**: Shapefiles, GeoJSON, satellite data
- **Levels**: Administrative boundaries, land use
- **Applications**: Mapping, spatial analysis, visualization

## Data Security and Compliance

### 🔒 Security Measures
- **Encryption**: AES-256 encryption at rest
- **Access Control**: Role-based permissions
- **Audit Logging**: All data access logged
- **Virus Scanning**: Automated malware detection

### 📋 Compliance
- **GDPR**: Data protection and privacy rights
- **Rwanda Data Protection**: Local regulations
- **NISR Standards**: Statistical confidentiality
- **Retention**: Automated data lifecycle management

## Data Quality Standards

### ✅ Validation Rules
- **Schema Validation**: Predefined data schemas
- **Quality Checks**: Missing values, outliers, duplicates
- **Geographic Validation**: Valid Rwanda administrative codes
- **Temporal Validation**: Date ranges and time series consistency

### 📊 Metadata Standards
- **Documentation**: Comprehensive data dictionaries
- **Lineage**: Full data processing history
- **Quality Scores**: Automated quality assessments
- **Usage Tracking**: Data access and usage patterns

## File Naming Conventions

### 📂 General Format
```
{data_type}_{geographic_level}_{time_period}_{version}.{extension}
```

### 📝 Examples
- `census_national_2022_v1.parquet` - National census data
- `economic_indicators_monthly_202310_v2.csv` - Economic indicators
- `household_survey_kigali_2023_final.xlsx` - Household survey
- `model_population_forecast_v3.pkl` - Population forecasting model

## Usage Guidelines

### 🎯 For Data Analysts
1. Always check data quality reports before analysis
2. Use processed data for analysis, not raw uploads
3. Document your analysis methodology
4. Follow NISR statistical standards

### 🤖 For ML Engineers
1. Use appropriate model folders for organization
2. Include model cards with performance metrics
3. Version control all model artifacts
4. Test models with validation datasets

### 📊 For Report Generators
1. Use standardized report templates
2. Include data sources and methodology
3. Follow NISR branding guidelines
4. Archive reports with proper metadata

## API Access

### 🔌 Programmatic Access
```python
# Example: Access processed census data
from nisr_analytics import DataManager

dm = DataManager()
census_data = dm.get_processed_data('census_national_2022')
```

### 🌐 REST API Endpoints
- `GET /api/v1/data/datasets` - List available datasets
- `GET /api/v1/data/{dataset_id}` - Get dataset metadata
- `POST /api/v1/data/upload` - Upload new data
- `GET /api/v1/models/{model_id}` - Get model information

## Monitoring and Maintenance

### 📈 Automated Monitoring
- **Storage Usage**: Alerts at 80% capacity
- **Data Quality**: Daily quality reports
- **Access Patterns**: Usage analytics
- **Performance**: Query performance monitoring

### 🔧 Maintenance Tasks
- **Daily**: Backup verification, quality checks
- **Weekly**: Performance optimization, cleanup
- **Monthly**: Capacity planning, security audit
- **Quarterly**: Data archival, compliance review

## Support and Documentation

### 📞 Contact Information
- **Technical Support**: analytics-support@nisr.gov.rw
- **Data Questions**: data-manager@nisr.gov.rw
- **System Issues**: it-support@nisr.gov.rw

### 📚 Additional Resources
- NISR Data Catalog: https://data.nisr.gov.rw
- API Documentation: https://api.nisr.gov.rw/docs
- User Guide: https://docs.nisr.gov.rw/analytics
- Training Materials: https://training.nisr.gov.rw

---

**Last Updated**: August 2024  
**Version**: 1.0  
**Maintained by**: NISR Analytics Team

# Database Save/Load Features - STL Analyzer

## New Features Added ✅

### **Save Database Button**
- **Location:** Database toolbar
- **Icon:** Save icon
- **Function:** Save current database to external file
- **Formats:** SQLite (.db), CSV (.csv), JSON (.json)

### **Load Database Button**  
- **Location:** Database toolbar
- **Icon:** Open icon
- **Function:** Load database from external file
- **Options:** Merge or Replace current database

### **Context Menu (Right-click on database entries)**
- **Load in 3D Viewer:** Load selected model in viewer
- **Open Containing Folder:** Open file location in explorer
- **Export Selected Entries:** Export selected rows to file
- **Remove from Database:** Remove selected entries

## How to Use

### **Saving Your Database**

1. **Click "Save Database" button** in the database toolbar
2. **Choose file format:**
   - `.db` - SQLite database (recommended for full compatibility)
   - `.csv` - Comma-separated values (for Excel/spreadsheet use)
   - `.json` - JSON format (for data exchange)
3. **Enter filename** (timestamp is auto-added)
4. **Click Save**

**Example filenames:**
- `stl_database_20250125_143022.db`
- `my_parts_collection.csv`
- `project_database.json`

### **Loading a Saved Database**

1. **Click "Load Database" button** in the database toolbar
2. **Select database file** (.db, .csv, or .json)
3. **Choose merge option:**
   - **Yes (Merge):** Add to current database (removes duplicates)
   - **No (Replace):** Replace current database entirely
   - **Cancel:** Cancel loading
4. **Database loads and view updates**

### **Using Context Menu (Right-click)**

**On any database entry:**
- **Load in 3D Viewer:** Instantly view the model
- **Open Containing Folder:** Navigate to file location
- **Export Selected:** Save selected entries to file
- **Remove from Database:** Delete entries (with confirmation)

## Use Cases

### **Project Management**
```
1. Scan project folder → Save as "project_alpha.db"
2. Scan another folder → Save as "project_beta.db"  
3. Load "project_alpha.db" when working on that project
4. Merge databases when combining projects
```

### **Backup & Sharing**
```
1. Save database as "backup_20250125.db"
2. Share database file with team members
3. Load shared databases from colleagues
4. Export specific entries as CSV for reports
```

### **Database Maintenance**
```
1. Right-click entries → Remove outdated files
2. Export important entries before cleanup
3. Load clean database after maintenance
4. Merge with new scans as needed
```

## File Formats

### **SQLite Database (.db)**
- **Best for:** Full compatibility with STL Analyzer
- **Contains:** All columns and data types preserved
- **Size:** Compact and efficient
- **Use when:** Sharing between STL Analyzer instances

### **CSV Files (.csv)**
- **Best for:** Excel, Google Sheets, data analysis
- **Contains:** All data in spreadsheet format
- **Size:** Larger than SQLite
- **Use when:** Need to analyze data in spreadsheet software

### **JSON Files (.json)**
- **Best for:** Data exchange, web applications
- **Contains:** Structured data format
- **Size:** Human-readable text format
- **Use when:** Integrating with other software

## Database Validation

### **Automatic Checks**
- ✅ Required columns present (filename, volume, surface_area)
- ✅ File format compatibility
- ✅ Data integrity validation
- ✅ Duplicate detection and handling

### **Error Handling**
- **Missing files:** Warns about invalid file paths
- **Corrupt data:** Shows detailed error messages
- **Format issues:** Suggests correct file formats
- **Empty databases:** Prevents loading empty files

## Performance

### **Loading Speed**
- **SQLite:** Fastest loading (optimized database format)
- **CSV:** Medium speed (text parsing required)
- **JSON:** Medium speed (structured parsing)

### **File Sizes (approximate)**
- **1000 entries:** SQLite ~200KB, CSV ~300KB, JSON ~400KB
- **10000 entries:** SQLite ~2MB, CSV ~3MB, JSON ~4MB

## Tips & Best Practices

### **Naming Convention**
```
project_name_YYYYMMDD.db
backup_YYYYMMDD_HHMM.db
shared_team_database.db
```

### **Regular Backups**
- Save database after major scans
- Keep dated backups before major changes
- Export important data as CSV for safety

### **Team Collaboration**
- Use SQLite format for sharing
- Include timestamp in filename
- Document what's included in shared databases

### **Database Cleanup**
- Use "Fix Database" to remove invalid entries
- Export important data before cleanup
- Remove entries for deleted files

## Troubleshooting

### **"Database is empty" error**
- File contains no valid data
- Check file format and content
- Try different file format

### **"Missing required columns" error**
- Database doesn't have filename/volume/surface_area columns
- File may not be STL Analyzer database
- Check if correct file was selected

### **"Failed to load database" error**
- File may be corrupted
- Check file permissions
- Try loading as different format

### **Merge vs Replace confusion**
- **Merge:** Keeps current + adds new (safe option)
- **Replace:** Deletes current + loads new (use carefully)
- **Cancel:** Stops loading (safe exit)

## Files Modified

1. `stl_analyzer/gui.py` - Added save/load functionality
2. `DATABASE_SAVE_LOAD_FEATURES.md` - This documentation

The database management system is now much more powerful and flexible! 🎉 
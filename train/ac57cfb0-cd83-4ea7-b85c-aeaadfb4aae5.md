**Product: [Product Name]**

---

**1. What is [Product Name]?**
A lightweight, cross-platform CLI tool designed for seamless [brief purpose, e.g., data parsing, automation, or task management].

---

**2. How do I install [Product Name]?**
```bash
# Linux/macOS
curl -L https://raw.githubusercontent.com/yourusername/[repo]/main/install.sh | bash

# Windows (PowerShell)
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/yourusername/[repo]/main/install.ps1" -OutFile install.ps1; .\install.ps1
```

---

**3. How do I use [Product Name]?**
```bash
[Product Name] <input_file> [-options] > output_file
```

---

**4. What flags are supported?**
```
-h, --help           Show this message
-i, --input         Input file path (default: stdin)
-o, --output        Output file path (default: stdout)
-v, --verbose       Enable debug logs
--json              Output in JSON format
--help              Display this help
```

---

**5. What are common use cases?**
- Parse CSV/JSON logs efficiently.
- Automate CLI workflows with minimal code.
- Integrate into scripts for batch processing.

---

**6. How do I get updates?**
```bash
[Product Name] --update
```

---
**7. What are the system requirements?**
- OS: Linux/macOS/Windows (64-bit)
- Min. CPU: x86_64 (ARM support in v2.0+)
- Min. RAM: 512MB

---
**8. How do I troubleshoot errors?**
Check logs with `--verbose`. Common issues: missing input, invalid flags, or permissions.
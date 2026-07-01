"""
静态文件服务索引页生成器
扫描 pages/ 目录，为每个子目录生成 index.html
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from html import escape


# ============ 模板常量 ============

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-hans" data-theme="dark">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#282828" id="theme-color-meta">
    <title>__TITLE__</title>
    <meta name="description" content="__DESCRIPTION__">
    <script>
        (function () {
            const saved = localStorage.getItem('theme-preference');
            const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            let resolvedTheme;
            let preference;

            if (saved === 'dark' || saved === 'light' || saved === 'auto') {
                preference = saved;
            } else {
                preference = 'auto';
            }

            if (preference === 'auto') {
                resolvedTheme = systemDark ? 'dark' : 'light';
            } else {
                resolvedTheme = preference;
            }

            document.documentElement.setAttribute('data-theme', resolvedTheme);
            document.documentElement.setAttribute('data-theme-preference', preference);

            const meta = document.getElementById('theme-color-meta');
            if (meta) {
                meta.setAttribute('content', resolvedTheme === 'dark' ? '#282828' : '#fbf1c7');
            }
        })();
    </script>
    <style>
        :root {
            --font-family:
                system-ui,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                Roboto,
                "Helvetica Neue",
                "Noto Sans",
                "Liberation Sans",
                "Apple Color Emoji",
                "Segoe UI Emoji",
                "Segoe UI Symbol",
                "Noto Color Emoji",
                "Source Han Sans VF",
                "Microsoft YaHei",
                "PingFang SC",
                "Hiragino Sans GB",
                "WenQuanYi Micro Hei",
                Arial,
                sans-serif;

            --font-family-monospace:
                "JetBrains Mono",
                "Liberation Mono",
                "SF Mono",
                Menlo,
                Monaco,
                Consolas,
                "Source Han Sans VF",
                "Microsoft YaHei",
                "PingFang SC",
                "Hiragino Sans GB",
                monospace;

            --transition-time: 0.2s;
            --transition-time-slow: 0.3s;
            --radius-sm: 2px;
            --radius-md: 4px;
            --breakpoint-sm: 576px;
            --breakpoint-md: 768px;
            --breakpoint-lg: 992px;
            --breakpoint-xl: 1200px;

            --color-primary: var(--green);
            --color-secondary: var(--yellow);
            --color-accent: var(--blue);
            --color-success: var(--green);
            --color-warning: var(--orange);
            --color-error: var(--red);
            --color-info: var(--blue);
            --text-primary: var(--fg1);
            --text-secondary: var(--fg-3);
            --text-muted: var(--fg-4);
            --text-inverse: var(--bg1);
            --bg-body: var(--bg);
            --bg-surface: var(--bg-s);
            --bg-overlay: #0000007F;
            --font-size-xs: 0.75rem;
            --font-size-sm: 0.875rem;
            --font-size-base: 1rem;
            --font-size-md: 1.125rem;
            --font-size-lg: 1.25rem;
            --font-size-xl: 1.5rem;
            font-size: 16px;
        }

        [data-theme="dark"] {
            color-scheme: dark;
            --bg_h: #1d2021;
            --bg: #282828;
            --bg_s: #32302f;
            --bg1: #3c3836;
            --bg2: #504945;
            --bg3: #665c54;
            --bg4: #7c6f64;
            --fg: #fbf1c7;
            --fg1: #ebdbb2;
            --fg2: #d5c4a1;
            --fg3: #bdae93;
            --fg4: #a89984;
            --red: #fb4934;
            --green: #b8bb26;
            --yellow: #fabd2f;
            --blue: #83a598;
            --purple: #d3869b;
            --aqua: #8ec07c;
            --gray: #928374;
            --orange: #fe8019;
            --red-dim: #cc2412;
            --green-dim: #98971a;
            --yellow-dim: #d79921;
            --blue-dim: #458588;
            --purple-dim: #b16286;
            --aqua-dim: #689d6a;
            --gray-dim: #a89984;
            --orange-dim: #d65d0e;
        }

        [data-theme="light"] {
            color-scheme: light;
            --bg_h: #f9f5d7;
            --bg: #fbf1c7;
            --bg_s: #f2e5bc;
            --bg1: #ebdbb2;
            --bg2: #d5c4a1;
            --bg3: #bdae93;
            --bg4: #a89984;
            --fg: #282828;
            --fg1: #3c3836;
            --fg2: #504945;
            --fg3: #665c54;
            --fg4: #7c6f64;
            --red: #9d0006;
            --green: #79740e;
            --yellow: #b57614;
            --blue: #076678;
            --purple: #8f3f71;
            --aqua: #427b58;
            --orange: #af3a03;
            --gray: #928374;
            --red-dim: #cc2412;
            --green-dim: #98971a;
            --yellow-dim: #d79921;
            --blue-dim: #458598;
            --purple-dim: #b16286;
            --aqua-dim: #689d6a;
            --orange-dim: #d65d0e;
            --gray-dim: #7c6f64;
        }

        a {
            color: var(--blue);
            text-decoration: none;
            transition: color 0.2s ease;
        }

        a:hover {
            color: var(--blue-dim);
        }

        a:visited {
            color: var(--blue);
        }

        a:active {
            color: var(--blue);
        }

        *,
        *::before,
        *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        html {
            scroll-behavior: smooth;
            -webkit-text-size-adjust: 100%;
        }

        body {
            font-family: var(--font-family);
            font-size: var(--font-size-sm);
            line-height: 1.5;
            color: var(--text-primary);
            background-color: var(--bg-body);
            min-height: 100vh;
        }

        .page-wrap {
            max-width: var(--breakpoint-xl);
            margin: 0 auto;
            padding: 1rem;
        }

        @media (max-width: 576px) {
            .page-wrap {
                padding: 0.5rem;
            }
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.5rem 0;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid var(--bg2);
        }

        .topbar-left {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            min-width: 0;
            flex: 1;
        }

        .topbar-right {
            display: flex;
            align-items: center;
            gap: 0.25rem;
            flex-shrink: 0;
        }

        .page-title {
            font-size: var(--font-size-base);
            font-weight: 600;
            color: var(--text-primary);
            white-space: nowrap;
        }

        .breadcrumb {
            display: flex;
            align-items: center;
            gap: 0.25rem;
            font-size: var(--font-size-sm);
            color: var(--text-muted);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .breadcrumb a {
            color: var(--blue);
        }

        .breadcrumb a:hover {
            color: var(--blue-dim);
        }

        .breadcrumb-sep {
            color: var(--bg3);
            user-select: none;
        }

        .breadcrumb-current {
            color: var(--text-primary);
            font-weight: 500;
        }

        .toolbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            padding: 0.375rem 0;
            margin-bottom: 0.25rem;
            flex-wrap: wrap;
        }

        .toolbar-left,
        .toolbar-right {
            display: flex;
            align-items: center;
            gap: 0.375rem;
        }

        .form-control {
            font-family: var(--font-family);
            font-size: var(--font-size-sm);
            line-height: 1.375;
            padding: 0.25rem 0.5rem;
            border: 1px solid var(--bg3);
            border-radius: var(--radius-sm);
            background-color: var(--bg1);
            color: var(--text-primary);
            outline: none;
            transition: border-color var(--transition-time) ease;
            min-height: 1.875rem;
        }

        .form-control:focus {
            border-color: var(--color-primary);
        }

        .form-control::placeholder {
            color: var(--text-muted);
        }

        select.form-control {
            cursor: pointer;
            padding-right: 1.5rem;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%23a89984' d='M0 0l5 6 5-6z'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 0.5rem center;
        }

        input[type="text"].form-control {
            width: 12rem;
        }

        @media (max-width: 576px) {
            input[type="text"].form-control {
                width: 100%;
            }
        }

        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.25rem;
            font-family: var(--font-family);
            font-size: var(--font-size-sm);
            font-weight: 500;
            line-height: 1.375;
            padding: 0.25rem 0.625rem;
            border: none;
            border-radius: var(--radius-sm);
            cursor: pointer;
            outline: none;
            transition: color var(--transition-time) ease, background-color var(--transition-time) ease;
            white-space: nowrap;
            min-height: 1.875rem;
        }

        .btn-text {
            color: var(--text-primary);
            background: transparent;
        }

        .btn-text:hover {
            color: var(--text-muted);
        }

        .btn-text:active {
            color: var(--text-primary);
        }

        .btn-text.active {
            color: var(--color-primary);
        }

        .btn-outline {
            color: var(--color-primary);
            background: transparent;
            border: 1px solid var(--color-primary);
        }

        .btn-outline:hover {
            background-color: var(--green-dim);
        }

        .btn-outline:active {
            background-color: transparent;
        }

        .file-table-wrap {
            border: 1px solid var(--bg2);
            border-radius: var(--radius-md);
            overflow: hidden;
        }

        .file-table {
            width: 100%;
            border-collapse: collapse;
            font-size: var(--font-size-sm);
        }

        .file-table thead {
            background-color: var(--bg1);
        }

        .file-table th {
            padding: 0.375rem 0.625rem;
            text-align: left;
            font-weight: 600;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--bg2);
            white-space: nowrap;
            cursor: pointer;
            user-select: none;
            transition: color var(--transition-time) ease;
        }

        .file-table th:hover {
            color: var(--text-primary);
        }

        .file-table th.sortable::after {
            content: "";
            display: inline-block;
            margin-left: 0.25rem;
            width: 0;
            height: 0;
            vertical-align: middle;
            border-left: 0.25rem solid transparent;
            border-right: 0.25rem solid transparent;
            border-top: 0.25rem solid var(--bg3);
            transition: border-color var(--transition-time) ease;
        }

        .file-table th.sort-asc::after {
            border-top: none;
            border-bottom: 0.25rem solid var(--color-primary);
        }

        .file-table th.sort-desc::after {
            border-top: 0.25rem solid var(--color-primary);
        }

        .file-table td {
            padding: 0.375rem 0.625rem;
            border-bottom: 1px solid var(--bg2);
            color: var(--text-primary);
            vertical-align: middle;
        }

        .file-table tbody tr {
            transition: background-color var(--transition-time) ease;
        }

        .file-table tbody tr:hover {
            background-color: var(--bg1);
        }

        .file-table tbody tr:last-child td {
            border-bottom: none;
        }

        .col-name {
            width: 100%;
            min-width: 8rem;
        }

        .file-link {
            display: flex;
            align-items: center;
            gap: 0.375rem;
            color: var(--text-primary);
            font-weight: 500;
        }

        .file-link:hover {
            color: var(--blue);
        }

        .file-type {
            display: inline-block;
            width: 1rem;
            height: 1rem;
            border-radius: var(--radius-sm);
            flex-shrink: 0;
            text-align: center;
            line-height: 1rem;
            font-size: var(--font-size-xs);
            font-family: var(--font-family-monospace);
            font-weight: 700;
        }

        .file-type-dir {
            color: var(--yellow);
            background-color: var(--yellow-dim);
        }

        .file-type-file {
            color: var(--aqua);
            background-color: var(--aqua-dim);
        }

        .col-size {
            width: 1%;
            white-space: nowrap;
            text-align: right;
        }

        .col-mtime {
            width: 1%;
            white-space: nowrap;
        }

        .col-action {
            width: 1%;
            white-space: nowrap;
            text-align: right;
        }

        .col-action .btn {
            padding: 0.125rem 0.375rem;
            min-height: auto;
            font-size: var(--font-size-xs);
        }

        .empty-state {
            padding: 2rem;
            text-align: center;
            color: var(--text-muted);
        }

        .footer-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            padding: 0.5rem 0;
            margin-top: 0.25rem;
            border-top: 1px solid var(--bg2);
            font-size: var(--font-size-xs);
            color: var(--text-muted);
            flex-wrap: wrap;
        }

        .footer-left,
        .footer-right {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .theme-switcher {
            position: relative;
        }

        .theme-dropdown {
            position: absolute;
            top: calc(100% + 0.25rem);
            right: 0;
            min-width: 7rem;
            background-color: var(--bg1);
            border: 1px solid var(--bg2);
            border-radius: var(--radius-md);
            overflow: hidden;
            z-index: 100;
            display: none;
        }

        .theme-dropdown.open {
            display: block;
        }

        .theme-option {
            display: block;
            width: 100%;
            padding: 0.375rem 0.625rem;
            font-size: var(--font-size-sm);
            color: var(--text-primary);
            background: none;
            border: none;
            text-align: left;
            cursor: pointer;
            transition: background-color var(--transition-time) ease;
        }

        .theme-option:hover {
            background-color: var(--bg2);
        }

        .theme-option.active {
            color: var(--color-primary);
        }

        @media (max-width: 768px) {
            .col-mtime {
                display: none;
            }
        }

        @media (max-width: 576px) {
            .col-size {
                display: none;
            }
            .toolbar {
                flex-direction: column;
                align-items: stretch;
            }
            .toolbar-left,
            .toolbar-right {
                justify-content: space-between;
            }
        }
    </style>
</head>
<body>
    <div class="page-wrap">
        <header class="topbar">
            <div class="topbar-left">
                <span class="page-title">文件索引</span>
                <nav class="breadcrumb" aria-label="面包屑导航">
__BREADCRUMB__
                </nav>
            </div>
            <div class="topbar-right">
                <div class="theme-switcher">
                    <button class="btn btn-text" id="theme-btn" type="button">主题</button>
                    <div class="theme-dropdown" id="theme-dropdown">
                        <button class="theme-option" data-value="auto">自动</button>
                        <button class="theme-option" data-value="light">浅色</button>
                        <button class="theme-option" data-value="dark">深色</button>
                    </div>
                </div>
            </div>
        </header>

        <div class="toolbar">
            <div class="toolbar-left">
                <input type="text" class="form-control" placeholder="搜索文件..." id="search-input">
                <select class="form-control" id="filter-type">
                    <option value="">全部类型</option>
                    <option value="dir">目录</option>
                    <option value="file">文件</option>
                </select>
            </div>
            <div class="toolbar-right">
                <button class="btn btn-text active" id="view-list" type="button">列表</button>
                <button class="btn btn-text" id="view-grid" type="button">网格</button>
                <select class="form-control" id="sort-select">
                    <option value="name">按名称</option>
                    <option value="size">按大小</option>
                    <option value="mtime">按时间</option>
                </select>
            </div>
        </div>

        <div class="file-table-wrap">
            <table class="file-table">
                <thead>
                    <tr>
                        <th class="col-name sortable sort-asc" data-sort="name">名称</th>
                        <th class="col-size sortable" data-sort="size">大小</th>
                        <th class="col-mtime sortable" data-sort="mtime">修改时间</th>
                        <th class="col-action"></th>
                    </tr>
                </thead>
                <tbody>
__FILE_ROWS__
                </tbody>
            </table>
        </div>

        <footer class="footer-bar">
            <div class="footer-left">
                <span>共 __TOTAL_COUNT__ 项</span>
                <span>|</span>
                <span>__DIR_COUNT__ 目录, __FILE_COUNT__ 文件</span>
            </div>
            <div class="footer-right">
                <span>静态文件服务</span>
            </div>
        </footer>
    </div>

    <script>
        (function () {
            const themeBtn = document.getElementById('theme-btn');
            const themeDropdown = document.getElementById('theme-dropdown');
            const themeOptions = themeDropdown.querySelectorAll('.theme-option');

            function getCurrentPreference() {
                return document.documentElement.getAttribute('data-theme-preference') || 'auto';
            }

            function setTheme(preference) {
                const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                let resolvedTheme;
                if (preference === 'auto') {
                    resolvedTheme = systemDark ? 'dark' : 'light';
                } else {
                    resolvedTheme = preference;
                }
                document.documentElement.setAttribute('data-theme', resolvedTheme);
                document.documentElement.setAttribute('data-theme-preference', preference);
                localStorage.setItem('theme-preference', preference);

                const meta = document.getElementById('theme-color-meta');
                if (meta) {
                    meta.setAttribute('content', resolvedTheme === 'dark' ? '#282828' : '#fbf1c7');
                }

                themeOptions.forEach(opt => {
                    opt.classList.toggle('active', opt.dataset.value === preference);
                });
            }

            themeBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                themeDropdown.classList.toggle('open');
            });

            themeOptions.forEach(opt => {
                opt.addEventListener('click', function () {
                    setTheme(this.dataset.value);
                    themeDropdown.classList.remove('open');
                });
            });

            document.addEventListener('click', function () {
                themeDropdown.classList.remove('open');
            });

            setTheme(getCurrentPreference());

            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () {
                if (getCurrentPreference() === 'auto') {
                    setTheme('auto');
                }
            });

            const sortHeaders = document.querySelectorAll('.file-table th.sortable');
            sortHeaders.forEach(th => {
                th.addEventListener('click', function () {
                    const sortKey = this.dataset.sort;
                    const isAsc = this.classList.contains('sort-asc');

                    sortHeaders.forEach(h => {
                        h.classList.remove('sort-asc', 'sort-desc');
                    });

                    if (isAsc) {
                        this.classList.add('sort-desc');
                    } else {
                        this.classList.add('sort-asc');
                    }
                });
            });

            const searchInput = document.getElementById('search-input');
            const filterType = document.getElementById('filter-type');
            const tbody = document.querySelector('.file-table tbody');
            const allRows = Array.from(tbody.querySelectorAll('tr'));

            function filterRows() {
                const query = searchInput.value.toLowerCase().trim();
                const type = filterType.value;

                allRows.forEach(row => {
                    const nameCell = row.querySelector('.col-name');
                    if (!nameCell) return;
                    const name = nameCell.textContent.toLowerCase();
                    const isDir = nameCell.querySelector('.file-type-dir') !== null;

                    let show = true;
                    if (query && !name.includes(query)) show = false;
                    if (type === 'dir' && !isDir) show = false;
                    if (type === 'file' && isDir) show = false;

                    row.style.display = show ? '' : 'none';
                });
            }

            searchInput.addEventListener('input', filterRows);
            filterType.addEventListener('change', filterRows);

            const viewListBtn = document.getElementById('view-list');
            const viewGridBtn = document.getElementById('view-grid');

            viewListBtn.addEventListener('click', function () {
                viewListBtn.classList.add('active');
                viewGridBtn.classList.remove('active');
            });

            viewGridBtn.addEventListener('click', function () {
                viewGridBtn.classList.add('active');
                viewListBtn.classList.remove('active');
            });
        })();
    </script>
</body>
</html>"""


# ============ 辅助函数 ============

def format_size(size_bytes: int) -> str:
    """将字节大小格式化为人类可读字符串"""
    if size_bytes < 0:
        return "-"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            if unit == "B":
                return f"{size_bytes} {unit}"
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def format_mtime(mtime: float) -> str:
    """将时间戳格式化为字符串"""
    dt = datetime.fromtimestamp(mtime)
    return dt.strftime("%Y-%m-%d %H:%M")


def build_breadcrumb(relative_path: str) -> str:
    """根据相对路径生成面包屑 HTML"""
    parts = [p for p in relative_path.replace("\\", "/").split("/") if p]
    lines = []
    lines.append('                    <a href="/">根目录</a>')
    accum = ""
    for i, part in enumerate(parts):
        lines.append('                    <span class="breadcrumb-sep">/</span>')
        accum = accum + part + "/"
        if i == len(parts) - 1:
            lines.append(f'                    <span class="breadcrumb-current">{part}</span>')
        else:
            lines.append(f'                    <a href="/{accum}">{part}</a>')
    return "\n".join(lines)


def build_file_rows(entries: list, has_parent: bool, current_rel_path: str) -> str:
    """生成文件列表行的 HTML"""
    lines = []
    if has_parent:
        lines.append(
            '                    <tr>\n'
            '                        <td class="col-name">\n'
            '                            <a href="../" class="file-link">\n'
            '                                <span class="file-type file-type-dir">D</span>\n'
            '                                <span>..</span>\n'
            '                            </a>\n'
            '                        </td>\n'
            '                        <td class="col-size">-</td>\n'
            '                        <td class="col-mtime">-</td>\n'
            '                        <td class="col-action"></td>\n'
            '                    </tr>'
        )
    for entry in entries:
        name = entry["name"]
        is_dir = entry["is_dir"]
        size_str = entry.get("size_str", "-")
        mtime_str = entry.get("mtime_str", "-")
        if is_dir:
            href = name + "/"
            type_class = "file-type-dir"
            type_label = "D"
            action_text = "打开"
        else:
            href = name
            type_class = "file-type-file"
            type_label = "F"
            action_text = "查看"
        name_escaped = escape(name)
        href_escaped = escape(href, quote=True)
        lines.append(
            f'                    <tr>\n'
            f'                        <td class="col-name">\n'
            f'                            <a href="{href_escaped}" class="file-link">\n'
            f'                                <span class="file-type {type_class}">{type_label}</span>\n'
            f'                                <span>{name_escaped}</span>\n'
            f'                            </a>\n'
            f'                        </td>\n'
            f'                        <td class="col-size">{size_str}</td>\n'
            f'                        <td class="col-mtime">{mtime_str}</td>\n'
            f'                        <td class="col-action">\n'
            f'                            <a href="{href_escaped}" class="btn btn-text">{action_text}</a>\n'
            f'                        </td>\n'
            f'                    </tr>'
        )
    return "\n".join(lines)


def scan_directory(dir_path: Path) -> list:
    """扫描目录，返回条目列表，目录在前按字母序，文件在后按字母序"""
    entries = []
    try:
        items = list(dir_path.iterdir())
        items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
        for item in items:
            if item.name.startswith(".") or item.name == "index.html":
                continue
            stat = item.stat()
            entry = {
                "name": item.name,
                "is_dir": item.is_dir(),
                "size": stat.st_size if item.is_file() else -1,
                "size_str": "-" if item.is_dir() else format_size(stat.st_size),
                "mtime": stat.st_mtime,
                "mtime_str": "-" if item.is_dir() else format_mtime(stat.st_mtime),
            }
            entries.append(entry)
    except PermissionError:
        pass
    return entries


def generate_index_html(target_dir: Path, pages_root: Path) -> str:
    """为单个目录生成 index.html 内容"""
    rel_path = target_dir.relative_to(pages_root).as_posix()
    if rel_path == ".":
        rel_path = ""
    entries = scan_directory(target_dir)
    has_parent = target_dir != pages_root
    breadcrumb = build_breadcrumb(rel_path)
    file_rows = build_file_rows(entries, has_parent, rel_path)
    dir_count = sum(1 for e in entries if e["is_dir"])
    file_count = sum(1 for e in entries if not e["is_dir"])
    total_count = len(entries) + (1 if has_parent else 0)
    title = f"文件索引 — {rel_path}" if rel_path else "文件索引"
    description = f"目录 {rel_path}" if rel_path else "根目录"

    html = HTML_TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__DESCRIPTION__", description)
    html = html.replace("__BREADCRUMB__", breadcrumb)
    html = html.replace("__FILE_ROWS__", file_rows)
    html = html.replace("__TOTAL_COUNT__", str(total_count))
    html = html.replace("__DIR_COUNT__", str(dir_count))
    html = html.replace("__FILE_COUNT__", str(file_count))
    return html


def find_all_dirs(pages_root: Path) -> list:
    """递归查找 pages/ 下所有目录"""
    dirs = [pages_root]
    for item in pages_root.rglob("*"):
        if item.is_dir():
            dirs.append(item)
    return sorted(dirs)


def main():
    """主入口"""
    import argparse
    parser = argparse.ArgumentParser(
        description="为 pages/ 目录下的所有子目录生成 index.html 索引页"
    )
    parser.add_argument(
        "--root",
        default="pages",
        help="pages 根目录路径 (默认: pages)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，不写入文件",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细信息",
    )
    args = parser.parse_args()
    pages_root = Path(args.root).resolve()
    if not pages_root.exists():
        print(f"错误: 目录不存在: {pages_root}", file=sys.stderr)
        sys.exit(1)
    if not pages_root.is_dir():
        print(f"错误: 不是目录: {pages_root}", file=sys.stderr)
        sys.exit(1)
    all_dirs = find_all_dirs(pages_root)
    generated = 0
    for target_dir in all_dirs:
        html = generate_index_html(target_dir, pages_root)
        output_path = target_dir / "index.html"
        rel = target_dir.relative_to(pages_root)
        if rel == Path("."):
            rel_str = "/"
        else:
            rel_str = "/" + rel.as_posix() + "/"
        if args.dry_run:
            print(f"[预览] {rel_str} -> {output_path}")
        else:
            output_path.write_text(html, encoding="utf-8")
            generated += 1
            if args.verbose:
                print(f"已生成: {rel_str} -> {output_path}")
    mode = "预览完成" if args.dry_run else "生成完成"
    print(f"{mode}，共 {len(all_dirs)} 个目录")
    if not args.dry_run:
        print(f"实际写入 {generated} 个 index.html")


if __name__ == "__main__":
    main()

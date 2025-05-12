"""
Text processing utilities including cleaning, entity extraction, and metadata processing.
"""

import os
import re
import time
from typing import List, Dict, Any, Set, Tuple, Optional
from datetime import datetime

# Global variables to track library availability
NLTK_AVAILABLE = False
SPACY_AVAILABLE = False
nlp = None

# Try to import NLTK
try:
    import nltk
    from nltk.corpus import stopwords, wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Try to import spaCy
try:
    import spacy
    # Try to load the model
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except:
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False


class TextProcessor:
    """Class for text processing and enhancement."""
    
    @staticmethod
    def clean_text(text: str, is_code: bool = False) -> str:
        """
        Clean and normalize text to improve embedding quality.
        
        Args:
            text: The text to clean
            is_code: Whether the text is code (affects cleaning process)
        """
        if not text:
            return ""
        
        if is_code:
            return TextProcessor.clean_code(text)
        
        # Standardize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('—', '-').replace('–', '-')
        
        # Remove excessive punctuation repetition
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Standardize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    @staticmethod
    def clean_code(text: str) -> str:
        """
        Clean code text while preserving structure and formatting.
        """
        if not text:
            return ""
        
        # Only standardize newlines for code
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove trailing whitespace from lines
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)
        
        # Remove multiple blank lines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    @staticmethod
    def extract_entities(text: str, is_code: bool = False, max_length: int = 15000) -> List[str]:
        """
        Extract named entities from text using spaCy if available.
        For code, extract identifiers, function names, class names, etc.

        Args:
            text: The text to extract entities from
            is_code: Whether the text is code
            max_length: Maximum text length to analyze for performance
        """
        if is_code:
            return TextProcessor.extract_code_entities(text, max_length)

        # For very large texts, limit analysis to a reasonable size
        # 15000 chars is enough to capture most important entities while improving performance
        analysis_text = text[:max_length] if len(text) > max_length else text

        if not SPACY_AVAILABLE or not analysis_text or not nlp:
            return []

        try:
            doc = nlp(analysis_text)
            entities = [ent.text for ent in doc.ents]

            # Return unique entities
            return list(set(entities))
        except Exception as e:
            print(f"Error during entity extraction: {e}")
            return []  # Return unique entities

    @staticmethod
    def extract_code_entities(text: str, max_length: int = 15000) -> List[str]:
        """
        Extract identifiers, function names, class names, and other
        important entities from code with balanced performance optimizations.
        """
        if not text:
            return []

        # For very large code files, limit analysis to a reasonable size
        analysis_text = text[:max_length] if len(text) > max_length else text

        entities = set()

        # Extract function and class definitions with comprehensive patterns
        patterns = [
            # Python
            r'def\s+(\w+)',
            r'class\s+(\w+)',
            # JavaScript/TypeScript
            r'function\s+(\w+)',
            r'class\s+(\w+)',
            r'(?:const|let|var)\s+(\w+)\s*=',
            # Java/C#
            r'(?:public|private|protected)?\s*(?:static)?\s*(?:\w+)\s+(\w+)\s*\(',
            r'(?:public|private|protected)?\s*class\s+(\w+)',
            # General
            r'import\s+(\w+)',
            r'from\s+(\w+)',
            r'package\s+(\w+)',
            r'namespace\s+(\w+)'
        ]

        import re
        for pattern in patterns:
            for match in re.finditer(pattern, analysis_text):
                entity = match.group(1)
                if entity and len(entity) > 1:  # Skip single-letter variables
                    entities.add(entity)

        # Extract camelCase and snake_case identifiers (important for code understanding)
        # But use reasonable limits to maintain performance
        camel_snake_samples = []

        # Take samples from different parts of the code
        sample_size = min(3000, len(analysis_text) // 3)
        if sample_size > 0:
            camel_snake_samples = [
                analysis_text[:sample_size],
                analysis_text[len(analysis_text)//2-sample_size//2:len(analysis_text)//2+sample_size//2],
                analysis_text[-sample_size:]
            ]

        # Extract camelCase and snake_case from samples
        identifier_patterns = [
            r'\b([a-z]+[A-Z][a-zA-Z0-9]*)\b',  # camelCase
            r'\b([a-z]+_[a-z][a-z0-9_]*)\b',   # snake_case
            r'\b([A-Z][a-zA-Z0-9]*)\b'         # PascalCase (classes/types)
        ]

        for sample in camel_snake_samples:
            for pattern in identifier_patterns:
                for match in re.finditer(pattern, sample):
                    entity = match.group(1)
                    if entity and len(entity) > 2 and not entity.startswith('_'):
                        # Skip common keywords
                        if entity.lower() not in [
                            'function', 'class', 'const', 'let', 'var', 'void', 'int',
                            'string', 'boolean', 'true', 'false', 'null', 'undefined',
                            'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case',
                            'break', 'continue', 'default', 'import', 'export', 'from'
                        ]:
                            entities.add(entity)

        # Extract commented terms that might be important
        # But limit to a reasonable number of matches to maintain performance
        comment_patterns = [
            r'#\s*(.+?)',        # Python single-line comment
            r'//\s*(.+?)',       # C-style single-line comment
            r'/\*\s*(.+?)\s*\*/'  # C-style multi-line comment
        ]

        comment_samples = camel_snake_samples[0:2]  # Use fewer samples for comments
        for sample in comment_samples:
            for pattern in comment_patterns:
                comment_count = 0
                for match in re.finditer(pattern, sample, re.MULTILINE):
                    comment = match.group(1).strip()
                    comment_count += 1

                    # Limit to a reasonable number of comments per pattern
                    if comment_count > 20:
                        break

                    # If spaCy is available, extract entities from comments
                    if SPACY_AVAILABLE and nlp:
                        try:
                            doc = nlp(comment[:200])  # Limit size for comments
                            for ent in doc.ents:
                                entities.add(ent.text)
                        except:
                            pass  # Continue even if NLP fails

        # Return unique entities, limited to a reasonable number
        entity_list = sorted(list(entities))
        if len(entity_list) > 100:
            # If we have too many entities, keep the most important ones
            # This is a reasonable limit that maintains quality while improving performance
            return entity_list

    @staticmethod
    def extract_language_from_path(file_path: str) -> str:
        """
        Determine the programming language from a file path.
        """
        ext = os.path.splitext(file_path)[1].lower()

        # Map extensions to language names
        language_map = {
            '.py': 'python',
            '.pyx': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.rs': 'rust',
            '.sh': 'bash',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.md': 'markdown',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss'
        }

        return language_map.get(ext, 'text')

    @staticmethod
    def is_code_file(file_path: str) -> bool:
        """
        Check if a file is a code file based on its extension.
        """
        code_extensions = {
            '.py', '.pyx', '.pyi', '.js', '.jsx', '.ts', '.tsx',
            '.java', '.kt', '.scala', '.c', '.cpp', '.cc', '.h',
            '.hpp', '.cs', '.go', '.rb', '.php', '.swift', '.rs',
            '.sh', '.bash', '.json', '.yaml', '.yml', '.toml',
            '.css', '.scss', '.less'
        }

        ext = os.path.splitext(file_path)[1].lower()
        return ext in code_extensions

    @staticmethod
    def extract_markdown_metadata(text: str, max_length: int = 20000) -> Dict[str, Any]:
        """
        Extract rich metadata from markdown files with balanced optimizations.

        Args:
            text: The markdown content
            max_length: Maximum text length to analyze for performance

        Returns:
            Dictionary of metadata including title, headers, links, etc.
        """
        if not text:
            return {}

        # For very large markdown files, limit analysis to a reasonable size
        analysis_text = text[:max_length] if len(text) > max_length else text

        metadata = {
            "title": None,
            "headers": [],
            "links": [],
            "code_blocks": [],
            "sections": [],
            "has_tables": False,
            "has_images": False,
            "estimated_reading_time": 0
        }

        import re

        # Extract title (first h1 header)
        title_match = re.search(r'^#\s+(.+)$', analysis_text, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        
        # Extract all headers with levels
        header_matches = re.finditer(r'^(#{1,6})\s+(.+)$', analysis_text, re.MULTILINE)
        header_count = 0
        for match in header_matches:
            level = len(match.group(1))
            content = match.group(2).strip()
            metadata["headers"].append({"level": level, "text": content})

            # Also add to sections for hierarchical representation
            section = {"level": level, "title": content, "start_pos": match.start()}
            metadata["sections"].append(section)
            
            header_count += 1
            # Limit headers for very large documents
            if header_count >= 50:
                break

        # Sort sections by position in document
        metadata["sections"] = sorted(metadata["sections"], key=lambda x: x["start_pos"])
        
        # Extract links - limit to first 100 links for performance
        link_matches = re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', analysis_text)
        link_count = 0
        for match in link_matches:
            link_text = match.group(1)
            url = match.group(2)
            metadata["links"].append({"text": link_text, "url": url})
            link_count += 1
            if link_count >= 100:
                break
        
        # Find code blocks with language - limit to first 30 blocks for performance
        code_block_matches = re.finditer(r'```(\w*)\n(.*?)\n```', analysis_text, re.DOTALL)
        block_count = 0
        for match in code_block_matches:
            language = match.group(1) or "text"
            code = match.group(2)
            metadata["code_blocks"].append({"language": language, "code": code[:100] + "..." if len(code) > 100 else code})
            block_count += 1
            if block_count >= 30:
                break
        
        # Check for tables
        if re.search(r'\|.*\|.*\n\|([\s\-:]+\|)+', analysis_text):
            metadata["has_tables"] = True

        # Check for images
        if re.search(r'!\[.*?\]\(.*?\)', analysis_text):
            metadata["has_images"] = True

        # Calculate estimated reading time (avg reading speed ~200-250 wpm)
        # Estimate word count based on a sample of the text for performance
        if len(text) > 10000:
            # Take three samples of 3000 chars each from beginning, middle, and end
            sample_text = text[:3000] + text[len(text)//2-1500:len(text)//2+1500] + text[-3000:]
            sample_words = len(re.findall(r'\b\w+\b', sample_text))
            # Extrapolate to full text
            words_per_char = sample_words / len(sample_text)
            word_count = int(len(text) * words_per_char)
        else:
            # Count words in the full text for smaller documents
            word_count = len(re.findall(r'\b\w+\b', text))

        metadata["word_count"] = word_count
        metadata["estimated_reading_time"] = max(1, round(word_count / 200))  # Minutes

        return metadata

    @staticmethod
    def extract_code_metadata(text: str, language: str, max_length: int = 15000) -> Dict[str, Any]:
        """
        Extract rich metadata from code files with balanced optimizations.

        Args:
            text: The code content
            language: The programming language
            max_length: Maximum text length to analyze for performance

        Returns:
            Dictionary of metadata including functions, classes, imports, etc.
        """
        if not text:
            return {"language": language}

        # For very large code files, limit analysis to a reasonable size
        analysis_text = text[:max_length] if len(text) > max_length else text

        metadata = {
            "language": language,
            "functions": [],
            "classes": [],
            "imports": [],
            "docstrings": [],
            "todos": [],
            "dependencies": [],
            "complexity_estimate": "low",
            "loc": 0,  # Lines of code
            "blank_lines": 0,
            "comment_lines": 0
        }

        # Count lines more efficiently
        lines = text.split("\n")
        metadata["loc"] = len(lines)

        # Count blank and comment lines with optimized patterns
        comment_patterns = {
            "python": r'^\s*#',
            "javascript": r'^\s*(//)|(^\s*/\*)|(\*/\s*$)',
            "typescript": r'^\s*(//)|(^\s*/\*)|(\*/\s*$)',
            "java": r'^\s*(//)|(^\s*/\*)|(\*/\s*$)',
            "c": r'^\s*(//)|(^\s*/\*)|(\*/\s*$)',
            "cpp": r'^\s*(//)|(^\s*/\*)|(\*/\s*$)',
            "csharp": r'^\s*(//)|(^\s*/\*)|(\*/\s*$)',
            "ruby": r'^\s*#',
            "php": r'^\s*(//)|(^\s*/\*)|(\*/\s*$)|(^\s*#)',
            "go": r'^\s*//',
            "rust": r'^\s*//',
        }

        import re
        comment_pattern = comment_patterns.get(language, r'^\s*(//|#|/\*)')

        # Only analyze a sample of lines for large files to improve performance
        if len(lines) > 1000:
            # Take samples from beginning, middle, and end
            sample_lines = lines[:300] + lines[len(lines)//2-150:len(lines)//2+150] + lines[-300:]
        else:
            sample_lines = lines

        for line in sample_lines:
            stripped = line.strip()
            if not stripped:
                metadata["blank_lines"] += 1
            elif re.match(comment_pattern, line):
                metadata["comment_lines"] += 1

        # Extrapolate for large files
        if len(lines) > 1000:
            sample_size = len(sample_lines)
            ratio = len(lines) / sample_size
            metadata["blank_lines"] = int(metadata["blank_lines"] * ratio)
            metadata["comment_lines"] = int(metadata["comment_lines"] * ratio)

        # Process by language
        if language == "python":
            metadata.update(TextProcessor._extract_python_metadata(analysis_text))
        elif language in ["javascript", "typescript"]:
            metadata.update(TextProcessor._extract_js_ts_metadata(analysis_text))
        elif language in ["java", "kotlin"]:
            metadata.update(TextProcessor._extract_java_metadata(analysis_text))
        elif language in ["c", "cpp", "csharp"]:
            metadata.update(TextProcessor._extract_c_family_metadata(analysis_text))
        else:
            # Generic extraction for other languages
            metadata.update(TextProcessor._extract_generic_code_metadata(analysis_text, language))

        # Estimate complexity based on several factors
        code_complexity = TextProcessor._estimate_code_complexity(metadata, text[:50000])  # Limit for performance
        metadata["complexity_estimate"] = code_complexity

        # Extract TODOs from a sample of the code
        todo_pattern = r'(TODO|FIXME|XXX|HACK)(?:\([^)]*\))?:?\s*([^\n]*)'
        todo_text = analysis_text[:30000]  # Limit TODO search to first 30k chars
        for match in re.finditer(todo_pattern, todo_text, re.IGNORECASE):
            todo_type = match.group(1).upper()
            todo_text = match.group(2).strip()
            metadata["todos"].append({"type": todo_type, "text": todo_text})

            # Limit number of TODOs for performance
            if len(metadata["todos"]) >= 20:
                break

        return metadata

    @staticmethod
    def _extract_python_metadata(text: str) -> Dict[str, Any]:
        """Extract metadata specific to Python code."""
        metadata = {
            "functions": [],
            "classes": [],
            "imports": [],
            "docstrings": [],
            "dependencies": []
        }

        # Extract imports
        import_patterns = [
            r'^import\s+([\w\.]+)(?:\s+as\s+(\w+))?',
            r'^from\s+([\w\.]+)\s+import\s+(.+?)(?:\s+as\s+(\w+))?$'
        ]
        
        for pattern in import_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                if match.group(0).startswith('from'):
                    module = match.group(1)
                    imported = match.group(2).strip()
                    alias = match.group(3) if len(match.groups()) > 2 else None
                    metadata["imports"].append({
                        "type": "from", 
                        "module": module, 
                        "imported": imported, 
                        "alias": alias
                    })
                    
                    # Add to dependencies list
                    root_module = module.split('.')[0]
                    if root_module and root_module not in metadata["dependencies"]:
                        metadata["dependencies"].append(root_module)
                else:
                    module = match.group(1)
                    alias = match.group(2) if len(match.groups()) > 1 else None
                    metadata["imports"].append({
                        "type": "import", 
                        "module": module, 
                        "alias": alias
                    })
                    
                    # Add to dependencies list
                    root_module = module.split('.')[0]
                    if root_module and root_module not in metadata["dependencies"]:
                        metadata["dependencies"].append(root_module)
        
        # Extract functions
        func_pattern = r'(?:^|\n)((?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[\w\[\],\s\.]*)?:)'
        for match in re.finditer(func_pattern, text, re.MULTILINE):
            full_def = match.group(1)
            func_name = match.group(2)
            
            # Look for docstring
            func_pos = match.end()
            docstring = ""
            triple_quote_pattern = r'"""(.+?)"""'
            docstring_match = re.search(triple_quote_pattern, text[func_pos:func_pos+500], re.DOTALL)
            if docstring_match:
                docstring = docstring_match.group(1).strip()
            
            metadata["functions"].append({
                "name": func_name,
                "signature": full_def,
                "has_docstring": bool(docstring),
                "is_async": full_def.strip().startswith("async")
            })
            
            if docstring:
                metadata["docstrings"].append({
                    "type": "function",
                    "name": func_name,
                    "docstring": docstring[:100] + "..." if len(docstring) > 100 else docstring
                })
        
        # Extract classes
        class_pattern = r'(?:^|\n)class\s+(\w+)(?:\s*\([^)]*\))?:'
        for match in re.finditer(class_pattern, text, re.MULTILINE):
            class_name = match.group(1)
            class_pos = match.end()
            
            # Look for docstring
            docstring = ""
            triple_quote_pattern = r'"""(.+?)"""'
            docstring_match = re.search(triple_quote_pattern, text[class_pos:class_pos+500], re.DOTALL)
            if docstring_match:
                docstring = docstring_match.group(1).strip()
            
            metadata["classes"].append({
                "name": class_name,
                "has_docstring": bool(docstring)
            })
            
            if docstring:
                metadata["docstrings"].append({
                    "type": "class",
                    "name": class_name,
                    "docstring": docstring[:100] + "..." if len(docstring) > 100 else docstring
                })
        
        # Look for module-level docstring at the top
        module_docstring_match = re.match(r'^"""(.+?)"""', text, re.DOTALL)
        if module_docstring_match:
            docstring = module_docstring_match.group(1).strip()
            metadata["docstrings"].append({
                "type": "module",
                "docstring": docstring[:100] + "..." if len(docstring) > 100 else docstring
            })
        
        return metadata
    
    @staticmethod
    def _extract_js_ts_metadata(text: str) -> Dict[str, Any]:
        """Extract metadata specific to JavaScript/TypeScript code."""
        metadata = {
            "functions": [],
            "classes": [],
            "imports": [],
            "exports": [],
            "docstrings": [],
            "dependencies": []
        }
        
        # Extract imports
        import_patterns = [
            r'import\s+\{([^}]+)\}\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'import\s+(\w+)\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'import\s+\*\s+as\s+(\w+)\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'require\([\'"]([^\'"]+)[\'"]\)'
        ]
        
        for pattern in import_patterns:
            for match in re.finditer(pattern, text):
                if pattern.startswith('import\\s+\\{'):
                    imports = [name.strip() for name in match.group(1).split(',')]
                    module = match.group(2)
                    metadata["imports"].append({
                        "type": "named", 
                        "module": module, 
                        "imported": imports
                    })
                    
                    # Add to dependencies
                    if module and not module.startswith('.') and module not in metadata["dependencies"]:
                        metadata["dependencies"].append(module)
                elif pattern.startswith('import\\s+\\*'):
                    alias = match.group(1)
                    module = match.group(2)
                    metadata["imports"].append({
                        "type": "namespace", 
                        "module": module, 
                        "alias": alias
                    })
                    
                    # Add to dependencies
                    if module and not module.startswith('.') and module not in metadata["dependencies"]:
                        metadata["dependencies"].append(module)
                elif 'require' in pattern:
                    module = match.group(1)
                    metadata["imports"].append({
                        "type": "require", 
                        "module": module
                    })
                    
                    # Add to dependencies
                    if module and not module.startswith('.') and module not in metadata["dependencies"]:
                        metadata["dependencies"].append(module)
                else:
                    default_import = match.group(1)
                    module = match.group(2)
                    metadata["imports"].append({
                        "type": "default", 
                        "module": module, 
                        "imported": default_import
                    })
                    
                    # Add to dependencies
                    if module and not module.startswith('.') and module not in metadata["dependencies"]:
                        metadata["dependencies"].append(module)
        
        # Extract exports
        export_patterns = [
            r'export\s+(?:default\s+)?(?:function|class)\s+(\w+)',
            r'export\s+(?:const|let|var)\s+(\w+)',
            r'export\s+\{([^}]+)\}'
        ]
        
        for pattern in export_patterns:
            for match in re.finditer(pattern, text):
                if pattern.endswith('\\}'):
                    exports = [name.strip() for name in match.group(1).split(',')]
                    metadata["exports"].append({
                        "type": "named",
                        "exported": exports
                    })
                else:
                    name = match.group(1)
                    export_type = "default" if "default" in match.group(0) else "named"
                    metadata["exports"].append({
                        "type": export_type,
                        "name": name
                    })
        
        # Extract functions (various formats)
        function_patterns = [
            # Regular functions
            r'function\s+(\w+)\s*\([^)]*\)\s*\{',
            # Arrow functions with explicit name
            r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>\s*[{]',
            # Class methods
            r'(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{'
        ]
        
        for pattern in function_patterns:
            for match in re.finditer(pattern, text):
                func_name = match.group(1)
                is_async = "async" in match.group(0)
                
                # Skip if this is part of a class method we'll capture separately
                if pattern.startswith('(?:async\\s+)?'):
                    # This is a potential class method, check if it's inside a class definition
                    pos = match.start()
                    class_check = text[:pos].rfind('class')
                    if class_check > -1 and text[class_check:pos].count('{') > text[class_check:pos].count('}'):
                        continue
                
                # Look for JSDoc comment
                func_start = match.start()
                docstring = ""
                jsdoc_pattern = r'/\*\*([^*]*\*+(?:[^/*][^*]*\*+)*/)'
                comments_before = text[max(0, func_start-500):func_start]
                jsdoc_match = re.search(jsdoc_pattern, comments_before, re.DOTALL)
                if jsdoc_match:
                    docstring = jsdoc_match.group(1).strip()
                
                metadata["functions"].append({
                    "name": func_name,
                    "has_docstring": bool(docstring),
                    "is_async": is_async
                })
                
                if docstring:
                    metadata["docstrings"].append({
                        "type": "function",
                        "name": func_name,
                        "docstring": docstring[:100] + "..." if len(docstring) > 100 else docstring
                    })
        
        # Extract classes
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{'
        for match in re.finditer(class_pattern, text):
            class_name = match.group(1)
            parent_class = match.group(2) if match.group(2) else None
            
            # Look for JSDoc comment
            class_start = match.start()
            docstring = ""
            jsdoc_pattern = r'/\*\*([^*]*\*+(?:[^/*][^*]*\*+)*/)'
            comments_before = text[max(0, class_start-500):class_start]
            jsdoc_match = re.search(jsdoc_pattern, comments_before, re.DOTALL)
            if jsdoc_match:
                docstring = jsdoc_match.group(1).strip()
            
            metadata["classes"].append({
                "name": class_name,
                "parent": parent_class,
                "has_docstring": bool(docstring)
            })
            
            if docstring:
                metadata["docstrings"].append({
                    "type": "class",
                    "name": class_name,
                    "docstring": docstring[:100] + "..." if len(docstring) > 100 else docstring
                })
        
        return metadata
    
    @staticmethod
    def _extract_java_metadata(text: str) -> Dict[str, Any]:
        """Extract metadata specific to Java/Kotlin code."""
        metadata = {
            "package": None,
            "imports": [],
            "classes": [],
            "interfaces": [],
            "methods": [],
            "javadoc": [],
            "annotations": []
        }
        
        # Extract package
        package_match = re.search(r'package\s+([^;]+);', text)
        if package_match:
            metadata["package"] = package_match.group(1).strip()
        
        # Extract imports
        for match in re.finditer(r'import\s+(?:static\s+)?([^;]+);', text):
            import_path = match.group(1).strip()
            is_static = "static" in match.group(0)
            wildcard = import_path.endswith(".*")
            
            metadata["imports"].append({
                "path": import_path,
                "is_static": is_static,
                "wildcard": wildcard
            })
        
        # Extract annotations
        annotation_pattern = r'@(\w+)(?:\([^)]*\))?'
        for match in re.finditer(annotation_pattern, text):
            annotation = match.group(1)
            metadata["annotations"].append(annotation)
        
        # Extract class definitions
        class_pattern = r'(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?\s*\{'
        for match in re.finditer(class_pattern, text):
            class_name = match.group(1)
            parent = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
            interfaces = []
            if len(match.groups()) > 2 and match.group(3):
                interfaces = [i.strip() for i in match.group(3).split(',')]
            
            # Look for JavaDoc comment
            class_start = match.start()
            javadoc = ""
            javadoc_pattern = r'/\*\*([^*]*\*+(?:[^/*][^*]*\*+)*/)'
            comments_before = text[max(0, class_start-500):class_start]
            javadoc_match = re.search(javadoc_pattern, comments_before, re.DOTALL)
            if javadoc_match:
                javadoc = javadoc_match.group(1).strip()
            
            class_info = {
                "name": class_name,
                "parent": parent,
                "interfaces": interfaces,
                "has_javadoc": bool(javadoc)
            }
            
            metadata["classes"].append(class_info)
            
            if javadoc:
                metadata["javadoc"].append({
                    "type": "class",
                    "name": class_name,
                    "text": javadoc[:100] + "..." if len(javadoc) > 100 else javadoc
                })
        
        # Extract interfaces
        interface_pattern = r'(?:public|private|protected)?\s*interface\s+(\w+)(?:\s+extends\s+([^{]+))?\s*\{'
        for match in re.finditer(interface_pattern, text):
            interface_name = match.group(1)
            extended = []
            if len(match.groups()) > 1 and match.group(2):
                extended = [i.strip() for i in match.group(2).split(',')]
            
            metadata["interfaces"].append({
                "name": interface_name,
                "extends": extended
            })
        
        # Extract methods
        method_pattern = r'(?:public|private|protected)?\s*(?:static|final|abstract)?\s*(?:<[^>]+>\s*)?(?:\w+)\s+(\w+)\s*\([^)]*\)(?:\s*throws\s+[^{]+)?(?:\s*\{|\s*;)'
        for match in re.finditer(method_pattern, text):
            method_name = match.group(1)
            is_abstract = "abstract" in match.group(0) or match.group(0).strip().endswith(";")
            
            # Look for JavaDoc comment
            method_start = match.start()
            javadoc = ""
            javadoc_pattern = r'/\*\*([^*]*\*+(?:[^/*][^*]*\*+)*/)'
            comments_before = text[max(0, method_start-500):method_start]
            javadoc_match = re.search(javadoc_pattern, comments_before, re.DOTALL)
            if javadoc_match:
                javadoc = javadoc_match.group(1).strip()
            
            metadata["methods"].append({
                "name": method_name,
                "is_abstract": is_abstract,
                "has_javadoc": bool(javadoc)
            })
            
            if javadoc:
                metadata["javadoc"].append({
                    "type": "method",
                    "name": method_name,
                    "text": javadoc[:100] + "..." if len(javadoc) > 100 else javadoc
                })
        
        return metadata
    
    @staticmethod
    def _extract_c_family_metadata(text: str) -> Dict[str, Any]:
        """Extract metadata specific to C/C++/C# code."""
        metadata = {
            "includes": [],
            "using_statements": [],
            "functions": [],
            "structs": [],
            "classes": [],
            "namespaces": [],
            "defines": [],
            "typedefs": []
        }
        
        # Extract includes
        for match in re.finditer(r'#\s*include\s+[<"]([^>"]+)[>"]', text):
            header = match.group(1)
            is_system = '<' in match.group(0)
            metadata["includes"].append({
                "header": header,
                "is_system": is_system
            })
        
        # Extract defines
        for match in re.finditer(r'#\s*define\s+(\w+)(?:\s+(.+))?, text, re.MULTILINE'):
            macro = match.group(1)
            value = match.group(2).strip() if match.group(2) else None
            metadata["defines"].append({
                "name": macro,
                "value": value
            })
        
        # Extract typedefs
        for match in re.finditer(r'typedef\s+(.+?)\s+(\w+);', text):
            original_type = match.group(1).strip()
            new_type = match.group(2)
            metadata["typedefs"].append({
                "original": original_type,
                "new": new_type
            })
        
        # Extract namespaces (C++)
        for match in re.finditer(r'namespace\s+(\w+)\s*\{', text):
            namespace = match.group(1)
            metadata["namespaces"].append(namespace)
        
        # Extract using statements (C#)
        for match in re.finditer(r'using\s+(?:(\w+)\s*=\s*)?([^;]+);', text):
            alias = match.group(1)
            namespace = match.group(2).strip()
            metadata["using_statements"].append({
                "namespace": namespace,
                "alias": alias
            })
        
        # Extract functions
        func_pattern = r'(?:(?:static|inline|virtual|extern|const|override|sealed|public|private|protected)\s+)*(?:\w+)(?:<[^>]+>)?\s+(?:\w+::)?(\w+)\s*\([^)]*\)(?:\s*const)?(?:\s*override)?(?:\s*=\s*0)?(?:\s*;\s*|\s*\{)'
        for match in re.finditer(func_pattern, text):
            func_name = match.group(1)
            is_definition = '{' in match.group(0)
            is_virtual = "virtual" in match.group(0) or "override" in match.group(0)
            is_abstract = "= 0" in match.group(0)
            
            # Extract documentation if present
            func_start = match.start()
            comment = ""
            doc_pattern = r'/\*\*([^*]*\*+(?:[^/*][^*]*\*+)*/)|\s*///(.+?)'
            comments_before = text[max(0, func_start-500):func_start]
            doc_match = re.search(doc_pattern, comments_before, re.DOTALL | re.MULTILINE)
            if doc_match:
                comment = (doc_match.group(1) or doc_match.group(2) or "").strip()
            
            metadata["functions"].append({
                "name": func_name,
                "is_definition": is_definition,
                "is_virtual": is_virtual,
                "is_abstract": is_abstract,
                "has_comment": bool(comment)
            })
        
        # Extract classes and structs
        class_pattern = r'(?:class|struct)\s+(\w+)(?:\s*:\s*(?:public|protected|private)\s+\w+(?:\s*,\s*(?:public|protected|private)\s+\w+)*)?\s*\{'
        for match in re.finditer(class_pattern, text):
            container_type = "class" if "class" in match.group(0) else "struct"
            name = match.group(1)
            
            # Extract inheritance
            inheritance = []
            inheritance_match = re.search(r':\s*(.+?)\s*\{', match.group(0))
            if inheritance_match:
                inheritance_text = inheritance_match.group(1)
                for parent in re.finditer(r'(?:public|protected|private)\s+(\w+)', inheritance_text):
                    inheritance.append({
                        "name": parent.group(1),
                        "access": parent.group(0).split()[0]
                    })
            
            container_info = {
                "name": name,
                "inheritance": inheritance
            }
            
            if container_type == "class":
                metadata["classes"].append(container_info)
            else:
                metadata["structs"].append(container_info)
        
        return metadata
    
    @staticmethod
    def _extract_generic_code_metadata(text: str, language: str) -> Dict[str, Any]:
        """Generic metadata extraction for any programming language."""
        metadata = {
            "functions": [],
            "classes": [],
            "imports": [],
            "comments": []
        }
        
        # Common patterns for many languages
        function_patterns = [
            r'function\s+(\w+)',  # JavaScript, PHP
            r'def\s+(\w+)',       # Python, Ruby
            r'sub\s+(\w+)',       # Perl
            r'func\s+(\w+)',      # Go
            r'fn\s+(\w+)',        # Rust
            r'\w+\s+(\w+)\s*\([^)]*\)\s*{'  # C-like
        ]
        
        for pattern in function_patterns:
            for match in re.finditer(pattern, text):
                func_name = match.group(1)
                metadata["functions"].append({
                    "name": func_name
                })
        
        # Class-like structures
        class_patterns = [
            r'class\s+(\w+)',  # Most OOP languages
            r'struct\s+(\w+)', # C, Go, Rust
            r'interface\s+(\w+)', # Java, Go, TypeScript
            r'type\s+(\w+)' # Go, TypeScript
        ]
        
        for pattern in class_patterns:
            for match in re.finditer(pattern, text):
                class_name = match.group(1)
                metadata["classes"].append({
                    "name": class_name
                })
        
        # Import/include statements
        import_patterns = [
            r'import\s+(.+?);',      # Java, Kotlin
            r'import\s+(.+?)',      # Python, JavaScript
            r'from\s+(.+?)\s+import', # Python
            r'#include\s+[<"](.+?)[>"]', # C/C++
            r'require\s+[\'"](.+?)[\'"]', # Ruby, Node.js
            r'use\s+(.+?);'         # PHP, Rust
        ]
        
        for pattern in import_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                imported = match.group(1).strip()
                metadata["imports"].append(imported)
        
        # Extract comments
        comment_patterns = [
            (r'//(.+?)', 'single-line'),   # C-style single line
            (r'#(.+?)', 'single-line'),    # Python, Ruby, Perl
            (r'/\*(.+?)\*/', 'multi-line'), # C-style multi-line
            (r'"""(.+?)"""', 'doc-string'), # Python
            (r"'''(.+?)'''", 'doc-string'), # Python (alternative)
            (r'/\*\*(.+?)\*/', 'javadoc')   # JavaDoc/JSDoc
        ]
        
        for pattern, comment_type in comment_patterns:
            for match in re.finditer(pattern, text, re.DOTALL | re.MULTILINE):
                comment_text = match.group(1).strip()
                if comment_text:
                    metadata["comments"].append({
                        "type": comment_type,
                        "text": comment_text[:100] + "..." if len(comment_text) > 100 else comment_text
                    })
        
        return metadata
    
    @staticmethod
    def _estimate_code_complexity(metadata: Dict[str, Any], text: str) -> str:
        """
        Estimate code complexity based on various metrics.
        Returns: "low", "medium", or "high"
        """
        # Initialize score
        complexity_score = 0
        
        # 1. Size-based metrics
        loc = metadata.get("loc", 0)
        if loc < 100:
            complexity_score += 0
        elif loc < 500:
            complexity_score += 1
        elif loc < 1000:
            complexity_score += 2
        else:
            complexity_score += 3
        
        # 2. Structure-based metrics
        num_functions = len(metadata.get("functions", []))
        num_classes = len(metadata.get("classes", []))
        
        if num_functions + num_classes > 20:
            complexity_score += 2
        elif num_functions + num_classes > 10:
            complexity_score += 1
        
        # 3. Nesting complexity
        # Count levels of indentation as a proxy for nesting
        lines = text.split('\n')
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        
        if max_indent > 24:  # More than 6 levels (assuming 4 spaces)
            complexity_score += 3
        elif max_indent > 16:  # More than 4 levels
            complexity_score += 2
        elif max_indent > 8:   # More than 2 levels
            complexity_score += 1
        
        # 4. Comment ratio
        comment_lines = metadata.get("comment_lines", 0)
        if loc > 0:
            comment_ratio = comment_lines / loc
            if comment_ratio < 0.1:  # Less than 10% comments
                complexity_score += 1
        
        # Determine complexity level
        if complexity_score >= 6:
            return "high"
        elif complexity_score >= 3:
            return "medium"
        else:
            return "low"
    
    @staticmethod
    def extract_file_metadata(file_path: str) -> Dict[str, Any]:
        """
        Extract general file metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of file metadata
        """
        try:
            file_stats = os.stat(file_path)
            
            # Get file extension and basename
            _, ext = os.path.splitext(file_path)
            basename = os.path.basename(file_path)
            
            # Get modification and creation times
            mtime = datetime.fromtimestamp(file_stats.st_mtime)
            ctime = datetime.fromtimestamp(file_stats.st_ctime)
            
            # Format times as strings
            mtime_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
            ctime_str = ctime.strftime("%Y-%m-%d %H:%M:%S")
            
            metadata = {
                "filename": basename,
                "extension": ext.lstrip('.').lower() if ext else "",
                "file_size_bytes": file_stats.st_size,
                "file_size_human": TextProcessor._human_readable_size(file_stats.st_size),
                "last_modified": mtime_str,
                "created": ctime_str,
                "source": os.path.abspath(file_path)
            }

            return metadata
        except Exception as e:
            print(f"Error extracting file metadata for {file_path}: {e}")
            return {
                "filename": os.path.basename(file_path),
                "path": file_path,
                "error": str(e)
            }
    
    @staticmethod
    def _human_readable_size(size_bytes: int) -> str:
        """Convert bytes to human-readable form."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        
        size_kb = size_bytes / 1024
        if size_kb < 1024:
            return f"{size_kb:.1f} KB"
        
        size_mb = size_kb / 1024
        if size_mb < 1024:
            return f"{size_mb:.1f} MB"
        
        size_gb = size_mb / 1024
        return f"{size_gb:.1f} GB"
    
    @staticmethod
    def expand_query(query: str) -> str:
        """
        Enhance query with synonyms or related terms.
        """
        if not NLTK_AVAILABLE or not query:
            return query
        
        try:
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            query_words = [word for word in query.split() if word.lower() not in stop_words]
            
            # Get synonyms for each meaningful word
            expanded_terms = set()
            for word in query_words:
                for syn in wordnet.synsets(word)[:2]:  # Limit to 2 synsets per word
                    for lemma in syn.lemmas()[:3]:  # Limit to 3 lemmas per synset
                        term = lemma.name().replace('_', ' ')
                        if term.lower() != word.lower():
                            expanded_terms.add(term)
            
            # Add up to 3 synonyms to the query
            top_terms = list(expanded_terms)[:3]
            if top_terms:
                return f"{query} {' '.join(top_terms)}"
            return query
        except Exception as e:
            print(f"Error expanding query: {e}")
            return query
    
    @staticmethod
    def expand_code_query(query: str) -> str:
        """
        Enhance code-related queries with programming terms and synonyms.
        """
        # Check if this seems to be a code-related query
        code_indicators = [
            'code', 'function', 'class', 'method', 'implementation',
            'algorithm', 'programming', 'script', 'module', 'library', 
            'api', 'interface', 'import', 'variable', 'syntax', 'error',
            'debug', 'compile', 'python', 'javascript', 'java', 'c++', 
            'typescript', 'html', 'css', 'ruby', 'go', 'rust'
        ]
        
        is_code_query = any(term in query.lower() for term in code_indicators)
        
        if not is_code_query:
            return query
        
        # Expand with code-specific terms
        language_expansions = {
            'python': ['def', 'class', 'import', 'function', 'module'],
            'javascript': ['function', 'const', 'let', 'var', 'async'],
            'java': ['public', 'class', 'static', 'void', 'method'],
            'c++': ['class', 'struct', 'template', 'namespace'],
            'typescript': ['interface', 'type', 'function', 'class']
        }
        
        # Detect potential language in query
        expanded_query = query
        for lang, terms in language_expansions.items():
            if lang in query.lower():
                # Add language-specific terms
                expanded_query += f" {' '.join(terms[:2])}"
                break
        
        return expanded_query
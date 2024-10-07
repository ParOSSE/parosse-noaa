"""
Helper functions to configure logging.
"""

import sys
import os
import logging
import shutil

# We create a hidden file to write the whole logs in case of a crash we can retrieve all log.
FULL_LOG_FILE = ".pbl_osse_full.log"

# Function needs to be call at the beggining of the file to run to display logs.
def configure_logger(exclude_keywords=[], level=2, log_file=None):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Provide a formatter that has the time, the file/class name and the message
    formatter = logging.Formatter('%(asctime)s %(name)s - %(message)s', '%Y-%m-%d %H:%M:%S')

    exclude_keyword_filter = ExcludeKeywordsFilter(exclude_keywords)

    if log_file is not None:
        main_handler = logging.FileHandler(log_file)
    else:
        main_handler = logging.StreamHandler(sys.stdout)

    if level == 0:
        main_handler.setLevel(logging.DEBUG)
    elif level == 1:
        main_handler.setLevel(logging.INFO)
    else:
        main_handler.setLevel(logging.WARNING)

    main_handler.addFilter(exclude_keyword_filter)
    main_handler.setFormatter(formatter)
    root.addHandler(main_handler)

     # Additional handler for displaying debug and info on crash
    crash_handler = logging.FileHandler(FULL_LOG_FILE)
    crash_handler.setLevel(logging.DEBUG)
    crash_handler.setFormatter(formatter)
    crash_handler.addFilter(exclude_keyword_filter)
    root.addHandler(crash_handler)


def crashed_log(log_file=None):
    if log_file is None:
        with open(FULL_LOG_FILE, 'r') as f:
            debug_logs = f.read()
        os.remove(FULL_LOG_FILE)
        print(f"Logs: \n {debug_logs}")
    else:
        shutil.move(FULL_LOG_FILE, log_file)


class ExcludeKeywordsFilter(logging.Filter):
    def __init__(self, exclude_keywords):
        super().__init__()
        self.exclude_keywords = exclude_keywords

    def filter(self, record):
        # Check if the log record contains any of the excluded keywords
        message = record.name + ' ' +record.getMessage()
        return not any(exclude_keyword in message for exclude_keyword in self.exclude_keywords)

from asyncio import CancelledError
from selenium import webdriver
from selenium.webdriver.edge.options import Options

from concurrent import futures
from concurrent.futures.thread import ThreadPoolExecutor
from threading import current_thread, Barrier

from time import sleep


class DriverPool:
    def __init__(self, max_drivers=10, *, headless=True, break_on_err=False):
        self.break_on_err = break_on_err
        self.drivers = {}
        self.tasks = set()
        self.options = Options()
        self.options.headless = headless
        self.executor = ThreadPoolExecutor(max_drivers, initializer=self.__init_driver)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.shutdown()

    def __init_driver(self):
        id = current_thread().ident
        print(f"[{id}] Initializing webdriver.")
        self.drivers[id] = webdriver.Edge(options=self.options)
        print(f"[{id}] Webdriver initialized.")

    def __shutdown_driver(self, barrier):
        barrier.wait()
        id = current_thread().ident
        print(f"[{id}] Shutting down webdriver.")
        driver = self.drivers[id]
        driver.quit()
        print(f"[{id}] Webdriver shut down.")
        
    def __loader(self, url, scraper, *args, **kwargs):
        id = current_thread().ident
        driver = self.drivers[id]
        print(f"[{id}] Loading URL: {url}")
        driver.get(url)
        print(f"[{id}] Finished loading.")
        scraper(driver, *args, **kwargs)

    def __on_finish(self, task):
        try:
            ex = task.exception()
        except futures.CancelledError:
            ex = None

        if ex is not None:
            id = current_thread().ident
            print(f"[{id}] Exception occurred during scrape: {ex}")

            if self.break_on_err:
                self.cancel()
            
        self.tasks.remove(task)

    def scrape(self, url, scraper, *args, **kwargs):
        task = self.executor.submit(self.__loader, url, scraper, *args, **kwargs)
        self.tasks.add(task)
        task.add_done_callback(self.__on_finish)

    def wait(self):
        futures.wait(self.tasks)

    def cancel(self):
        for task in list(self.tasks):
            task.cancel()

    def shutdown(self):
        try:
            self.wait()
        finally:
            self.wait()

            # Deliver a poison pill to all workers
            num_workers = len(self.drivers)
            barrier = Barrier(num_workers)
            shutdown_tasks = [self.executor.submit(self.__shutdown_driver, barrier) for i in range(num_workers)]
            futures.wait(shutdown_tasks)

            # Shutdown the executor
            self.executor.shutdown()


class WindowPool:
    def __init__(self, max_windows=10, *, headless=True, break_on_err=False):
        self.break_on_err = break_on_err
        self.windows = {}
        self.tasks = set()
        self.options = Options()
        self.options.headless = headless
        self.driver = None
        self.driver_executor = ThreadPoolExecutor(1, initializer=self.__init_driver)
        self.executor = ThreadPoolExecutor(max_windows, initializer=self.__init_window)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.shutdown()

    def __init_driver(self):
        self.driver = webdriver.Edge(options=self.options)

    def _driver_invoke(self, fn, *args, **kwargs):
        task = self.driver_executor.submit(fn, *args, **kwargs)
        return task.result()

    def __init_window_sync(self, id):
        # Open a new tab if needed
        if len(self.driver.window_handles) == len(self.windows):
            self.driver.switch_to.new_window('tab')

        # Find new tab
        window_handle_set = set(self.driver.window_handles) - set(self.windows.values())
        window_handle = list(window_handle_set)[0]
        self.windows[id] = window_handle

    def __init_window(self):
        id = current_thread().ident
        print(f"[{id}] Initializing window.")
        
        self._driver_invoke(self.__init_window_sync, id)
        
        print(f"[{id}] Window initialized.")
        
    def __loader_sync_load(self, id, url):
        window_handle = self.windows[id]
        self.driver.switch_to.window(window_handle)
        self.driver.execute_script(f"""
        setTimeout(function() {{ 
            window.location.href = '{url}';
        }}, 10);
        """)

    def __loader_sync_isloaded(self, id):
        window_handle = self.windows[id]
        self.driver.switch_to.window(window_handle)
        return self.driver.execute_script("return document.readyState === 'complete';")
        
    def __loader_sync_scrape(self, id, scraper, *args, **kwargs):
        window_handle = self.windows[id]
        self.driver.switch_to.window(window_handle)
        scraper(self.driver, *args, **kwargs)

    def __loader(self, url, scraper, *args, **kwargs):
        id = current_thread().ident
        self._driver_invoke(self.__loader_sync_load, id, url)
        print(f"[{id}] Loading URL: {url}")
        for i in range(20):
            sleep(0.5)
            if self._driver_invoke(self.__loader_sync_isloaded, id):
                break
        print(f"[{id}] Finished loading.")
        self._driver_invoke(self.__loader_sync_scrape, id, scraper, *args, **kwargs)

    def __on_finish(self, task):
        self.tasks.remove(task)

        try:
            ex = task.exception()
        except futures.CancelledError:
            ex = None

        if ex is not None:
            id = current_thread().ident
            print(f"[{id}] Exception occurred during scrape: {ex}")

            if self.break_on_err:
                self.cancel()

    def scrape(self, url, scraper, *args, **kwargs):
        task = self.executor.submit(self.__loader, url, scraper, *args, **kwargs)
        self.tasks.add(task)
        task.add_done_callback(self.__on_finish)

    def wait(self):
        futures.wait(self.tasks)

    def cancel(self):
        for task in list(self.tasks):
            task.cancel()

    def shutdown(self):
        try:
            self.wait()
        finally:
            # Shutdown the task executor
            self.executor.shutdown()

            # Send the shutdown command to the driver
            if self.driver is not None:
                self._driver_invoke(self.driver.quit)

            # Shutdown the driver executor
            self.driver_executor.shutdown() 
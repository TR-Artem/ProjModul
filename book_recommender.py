import sys
import pandas as pd
import numpy as np
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QLabel, QComboBox, QPushButton, QListWidget, 
                             QMessageBox, QHBoxLayout, QSlider, QListWidgetItem,
                             QFrame, QScrollArea)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon, QFont
import urllib.request

# Список русских стоп-слов
RUSSIAN_STOP_WORDS = [
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 
    'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 
    'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от'
]

# Класс для фоновой загрузки изображений
class ImageLoaderThread(QThread):
    loaded = pyqtSignal(QPixmap)
    error = pyqtSignal()

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        try:
            with urllib.request.urlopen(self.url) as response:
                data = response.read()
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            self.loaded.emit(pixmap.scaled(120, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except:
            self.error.emit()

# Класс для вычисления рекомендаций в фоне
class RecommendationWorker(QThread):
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, recommender, book_id, method, top_n):
        super().__init__()
        self.recommender = recommender
        self.book_id = book_id
        self.method = method
        self.top_n = top_n

    def run(self):
        try:
            if self.method == 'knn':
                result = self.recommender.get_knn_recommendations(self.book_id, self.top_n)
            else:
                result = self.recommender.get_recommendations(self.book_id, self.method, self.top_n)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

# Виджет элемента книги в списке
class BookItemWidget(QWidget):
    def __init__(self, book_data, book_images, parent=None):
        super().__init__(parent)
        self.book_data = book_data
        self.book_images = book_images
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # Обложка книги
        self.cover_label = QLabel()
        self.cover_label.setFixedSize(120, 180)
        self.cover_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.cover_label)

        # Информация о книге
        info_layout = QVBoxLayout()
        info_layout.setSpacing(8)

        self.title_label = QLabel(f"<b>{self.book_data['title']}</b>")
        self.title_label.setStyleSheet("font-size: 16px; color: #bb86fc;")
        info_layout.addWidget(self.title_label)

        self.author_label = QLabel(f"Автор: {self.book_data['author']}")
        info_layout.addWidget(self.author_label)

        self.genre_label = QLabel(f"Жанр: {self.book_data['genre']}")
        info_layout.addWidget(self.genre_label)

        # Кнопка "Подробнее"
        details_btn = QPushButton("Подробнее")
        details_btn.setFixedSize(120, 30)
        details_btn.clicked.connect(self.show_details)
        info_layout.addWidget(details_btn, alignment=Qt.AlignRight)

        layout.addLayout(info_layout, stretch=1)
        self.setLayout(layout)

        # Загрузка изображения
        self.load_cover()

    def load_cover(self):
        if self.book_data['book_id'] in self.book_images:
            self.thread = ImageLoaderThread(self.book_images[self.book_data['book_id']])
            self.thread.loaded.connect(self.set_cover_image)
            self.thread.error.connect(self.show_placeholder)
            self.thread.start()
        else:
            self.show_placeholder()

    def set_cover_image(self, pixmap):
        self.cover_label.setPixmap(pixmap)

    def show_placeholder(self):
        self.cover_label.setText("Нет обложки")

    def show_details(self):
        try:
            image_data = urllib.request.urlopen(self.book_images[self.book_data['book_id']]).read()
            self.details_window = BookDetailsWindow(self.book_data, image_data, self)
            self.details_window.show()
        except:
            QMessageBox.warning(self, "Ошибка", "Не удалось загрузить детали книги")

# Окно с деталями книги
class BookDetailsWindow(QWidget):
    def __init__(self, book_data, image_data, parent=None):
        super().__init__(parent)
        self.setup_ui(book_data, image_data)

    def setup_ui(self, book_data, image_data):
        self.setWindowTitle("Описание книги")
        self.setGeometry(200, 200, 600, 500)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Заголовок
        title_label = QLabel(f"<h2>{book_data['title']}</h2>")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Обложка
        cover_label = QLabel()
        pixmap = QPixmap()
        pixmap.loadFromData(image_data)
        cover_label.setPixmap(pixmap.scaled(250, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        cover_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(cover_label)

        # Автор и жанр
        author_label = QLabel(f"<b>Автор:</b> {book_data['author']}")
        layout.addWidget(author_label)

        genre_label = QLabel(f"<b>Жанр:</b> {book_data['genre']}")
        layout.addWidget(genre_label)

        # Разделитель
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Описание с прокруткой
        desc_label = QLabel(book_data['description'])
        desc_label.setWordWrap(True)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(desc_label)
        layout.addWidget(scroll)

        self.setLayout(layout)

# Класс рекомендательной системы
class AdvancedBookRecommender:
    def __init__(self):
        self.books, self.ratings = self.load_extended_data()
        self.user_book_matrix = None
        self.content_similarity = None
        self.collab_similarity = None
        self.hybrid_model = None
        self.knn_model = None
        self.book_images = self.load_book_images()
        self.prepare_models()
        self._cache = {}

    def load_book_images(self):
        return {
            1: "https://m.media-amazon.com/images/I/71kxa1-0mfL._AC_UF1000,1000_QL80_.jpg",
            # ... остальные URL изображений ...
            15: "https://m.media-amazon.com/images/I/91DfS2k6BLL._AC_UF1000,1000_QL80_.jpg"
        }

    def load_extended_data(self):
        books = pd.DataFrame({
            'book_id': range(1, 16),
            'title': ['1984', 'Гарри Поттер и философский камень', 'Мастер и Маргарита', 
                     'Преступление и наказание', 'Три товарища', 'Маленький принц',
                     'Атлант расправил плечи', 'Шерлок Холмс', 'Война и мир', 'Горе от ума',
                     '451 градус по Фаренгейту', 'Убить пересмешника', 'Властелин колец',
                     'Над пропастью во ржи', 'Анна Каренина'],
            'author': ['Джордж Оруэлл', 'Дж. К. Роулинг', 'Михаил Булгаков',
                      'Федор Достоевский', 'Эрих Мария Ремарк', 'Антуан де Сент-Экзюпери',
                      'Айн Рэнд', 'Артур Конан Дойл', 'Лев Толстой', 'Александр Грибоедов',
                      'Рэй Брэдбери', 'Харпер Ли', 'Дж. Р. Р. Толкин',
                      'Джером Сэлинджер', 'Лев Толстой'],
            'genre': ['Антиутопия', 'Фэнтези', 'Магический реализм', 'Классика', 'Роман', 
                     'Философская сказка', 'Роман', 'Детектив', 'Исторический роман', 'Комедия',
                     'Антиутопия', 'Роман', 'Фэнтези', 'Роман', 'Классика'],
            'description': [
                'Роман-антиутопия о тоталитарном обществе',
                'Первая книга о юном волшебнике Гарри Поттере',
                'Мистический роман о дьяволе, посещающем Москву',
                'История бывшего студента, совершившего убийство',
                'История дружбы трех товарищей в послевоенной Германии',
                'Философская сказка о маленьком принце с другой планеты',
                'Роман о роли разума в жизни человека и общества',
                'Сборник рассказов о знаменитом детективе',
                'Эпический роман о войне с Наполеоном',
                'Сатирическая комедия о нравах дворянства',
                'Антиутопия о мире, где книги под запретом',
                'История о расовой несправедливости в Америке',
                'Эпическая фэнтези-сага о борьбе за Кольцо Всевластья',
                'История подростка, переживающего экзистенциальный кризис',
                'Трагическая история любви замужней женщины'
            ]
        })
        
        ratings = pd.DataFrame({
            'user_id': np.random.randint(1, 100, 500),
            'book_id': np.random.choice(range(1, 16), 500),
            'rating': np.random.randint(1, 6, 500)
        })
        
        return books, ratings

    def prepare_models(self):
        self.books['metadata'] = self.books['genre'] + ' ' + self.books['author'] + ' ' + self.books['description']
        
        tfidf = TfidfVectorizer(stop_words=RUSSIAN_STOP_WORDS, max_features=5000)
        tfidf_matrix = tfidf.fit_transform(self.books['metadata'])
        self.content_similarity = cosine_similarity(tfidf_matrix)
        
        self.user_book_matrix = self.ratings.pivot_table(
            index='user_id', columns='book_id', values='rating', fill_value=0)
        
        svd = TruncatedSVD(n_components=10, random_state=42)
        reduced_matrix = svd.fit_transform(self.user_book_matrix.T)
        self.collab_similarity = cosine_similarity(reduced_matrix)
        
        self.hybrid_model = 0.5 * self.content_similarity + 0.5 * self.collab_similarity
        self.knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
        self.knn_model.fit(reduced_matrix)

    def get_recommendations(self, book_id, method='hybrid', top_n=5):
        cache_key = (book_id, method, top_n)
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        if method == 'content':
            similarity_scores = self.content_similarity[book_id - 1]
        elif method == 'collab':
            similarity_scores = self.collab_similarity[book_id - 1]
        else:
            similarity_scores = self.hybrid_model[book_id - 1]
        
        similar_indices = np.argsort(similarity_scores)[-top_n-1:-1][::-1]
        result = self.books.iloc[similar_indices]
        self._cache[cache_key] = result
        return result

    def get_knn_recommendations(self, book_id, top_n=5):
        distances, indices = self.knn_model.kneighbors(
            [self.collab_similarity[book_id - 1]], n_neighbors=top_n+1)
        return self.books.iloc[indices[0][1:]]

# Главное окно приложения
class RecommenderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Книжный рекомендатель")
        self.setGeometry(100, 100, 1000, 800)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        central_widget.setLayout(main_layout)

        # Заголовок
        title_label = QLabel("📖 Книжный рекомендатель")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #bb86fc; padding: 10px;")
        main_layout.addWidget(title_label, alignment=Qt.AlignCenter)

        # Выбор книги
        book_layout = QHBoxLayout()
        book_label = QLabel("Выберите книгу:")
        book_label.setStyleSheet("font-size: 14px;")
        book_layout.addWidget(book_label)

        self.book_combo = QComboBox()
        book_layout.addWidget(self.book_combo)
        main_layout.addLayout(book_layout)

        # Метод рекомендаций
        method_layout = QHBoxLayout()
        method_label = QLabel("Метод рекомендаций:")
        method_label.setStyleSheet("font-size: 14px;")
        method_layout.addWidget(method_label)

        self.method_combo = QComboBox()
        self.method_combo.addItem("Гибридный (лучший)", "hybrid")
        self.method_combo.addItem("По содержанию", "content")
        self.method_combo.addItem("По оценкам пользователей", "collab")
        self.method_combo.addItem("KNN-рекомендации", "knn")
        method_layout.addWidget(self.method_combo)
        main_layout.addLayout(method_layout)

        # Количество рекомендаций
        count_layout = QHBoxLayout()
        count_label = QLabel("Количество рекомендаций:")
        count_label.setStyleSheet("font-size: 14px;")
        count_layout.addWidget(count_label)

        self.count_slider = QSlider(Qt.Horizontal)
        self.count_slider.setMinimum(3)
        self.count_slider.setMaximum(10)
        self.count_slider.setValue(5)
        count_layout.addWidget(self.count_slider)

        self.count_label = QLabel("5")
        self.count_label.setStyleSheet("font-size: 14px;")
        count_layout.addWidget(self.count_label)
        main_layout.addLayout(count_layout)

        # Кнопка получения рекомендаций
        self.recommend_btn = QPushButton("Получить рекомендации")
        self.recommend_btn.setStyleSheet("""
            QPushButton {
                background-color: #bb86fc;
                color: #000000;
                font-weight: bold;
                border-radius: 4px;
                padding: 10px 20px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #9a67ea;
            }
            QPushButton:pressed {
                background-color: #7e57c2;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #aaa;
            }
        """)
        self.recommend_btn.clicked.connect(self.show_recommendations)
        main_layout.addWidget(self.recommend_btn)

        # Список рекомендаций
        recommendations_label = QLabel("Рекомендуемые книги:")
        recommendations_label.setStyleSheet("font-size: 14px;")
        main_layout.addWidget(recommendations_label)

        self.recommendations_list = QListWidget()
        self.recommendations_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QListWidget::item {
                height: 200px;
            }
        """)
        main_layout.addWidget(self.recommendations_list)

        # Инициализация данных
        self.recommender = AdvancedBookRecommender()
        self.init_data()

        # Подключение сигналов
        self.count_slider.valueChanged.connect(lambda: self.count_label.setText(str(self.count_slider.value())))

    def init_data(self):
        # Заполнение выпадающего списка книгами
        for _, row in self.recommender.books.iterrows():
            self.book_combo.addItem(f"{row['title']} - {row['author']}", row['book_id'])

    def show_recommendations(self):
        # Блокируем кнопку на время выполнения
        self.recommend_btn.setEnabled(False)
        self.recommend_btn.setText("Обработка...")
        self.recommendations_list.clear()

        # Получаем параметры
        book_id = self.book_combo.currentData()
        method = self.method_combo.currentData()
        top_n = self.count_slider.value()

        # Запускаем worker в отдельном потоке
        self.worker = RecommendationWorker(self.recommender, book_id, method, top_n)
        self.worker.finished.connect(self.display_recommendations)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(lambda: self.recommend_btn.setEnabled(True))
        self.worker.error.connect(lambda: self.recommend_btn.setEnabled(True))
        self.worker.start()

    def display_recommendations(self, recommendations):
        self.recommend_btn.setText("Получить рекомендации")
        
        if recommendations.empty:
            QMessageBox.information(self, "Информация", "Рекомендации не найдены")
            return

        for _, row in recommendations.iterrows():
            item = QListWidgetItem()
            widget = BookItemWidget(row, self.recommender.book_images, self)
            item.setSizeHint(widget.sizeHint())
            self.recommendations_list.addItem(item)
            self.recommendations_list.setItemWidget(item, widget)

    def show_error(self, error_msg):
        self.recommend_btn.setText("Получить рекомендации")
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{error_msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Установка темной палитры
    palette = app.palette()
    palette.setColor(palette.Window, QColor(53, 53, 53))
    palette.setColor(palette.WindowText, Qt.white)
    palette.setColor(palette.Base, QColor(25, 25, 25))
    palette.setColor(palette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ToolTipBase, Qt.white)
    palette.setColor(palette.ToolTipText, Qt.white)
    palette.setColor(palette.Text, Qt.white)
    palette.setColor(palette.Button, QColor(53, 53, 53))
    palette.setColor(palette.ButtonText, Qt.white)
    palette.setColor(palette.BrightText, Qt.red)
    palette.setColor(palette.Link, QColor(42, 130, 218))
    palette.setColor(palette.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = RecommenderApp()
    window.show()
    sys.exit(app.exec_())
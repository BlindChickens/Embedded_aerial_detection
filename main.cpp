#include "main.h"

QApplication *app;
MainWindow *window;



int main(int argc, char *argv[])
{
    app = new QApplication (argc, argv);
    window = new MainWindow();

    window->show();

    return app->exec();
}

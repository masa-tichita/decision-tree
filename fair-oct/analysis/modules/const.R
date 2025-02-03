source("analysis/modules/lib.R")
library(Dict)


common_format_tags_data <- function(data_path) {
  ## データの読み込み
  df_raw <- read.csv(data_path)
  df <- df_raw %>%
    group_by(workName) %>%
    arrange(startAt) %>%
    mutate(
      startDatetime = as.POSIXct(startAt, origin = "1970-01-01"),
      endDatetime = as.POSIXct(endAt, origin = "1970-01-01"),
      date = as_date(startDatetime),
      # 作業時間
      duration = endDatetime - startDatetime,
      # 人時生産性
      humanProductivity = 3600 / (meanCycleTime * meanWorkerNum),
      # 段取り時間
      changeover = machineStartAt - lag(machineEndAt),
    ) %>%
    drop_na(humanProductivity) %>%
    drop_na(changeover)
  return(df)
}


# conf <- read_yaml("config.yml")
# term <- conf$data
# factory <- conf$factory

# read_csv_in_local <- function (csv_prefix_list, folder_path="data/") {
#   output <- data.frame()
#   data_dict <- list()
#   # print(getwd())
#   absolute_path <- paste0(getwd(), "/", folder_path, term)
#   files <- list.files(absolute_path)
#   for (file in files) {
#     print(file)
#     for (csv_prefix in csv_prefix_list) {
#       print(csv_prefix)
#       if (grepl(csv_prefix, file)) {
#         print("hello")
#         print(paste0(absolute_path, "/", file, ".csv"))
#         output <- read.csv(paste0(absolute_path, "/", file, ".csv"))
#         data_dict[[file]] <- output
#       }
#     }
#   }
#   return (data_dict)
# }
read_csv_in_local <- function(csv_prefix_list, folder_path = "data/") {
  # 名前付きリストを使用（Dictの代わりに）
  data_dict <- list()

  # 絶対パスの構築（termは削除）
  absolute_path <- file.path(getwd(), folder_path, term)
  print(paste("Absolute Path:", absolute_path))

  # フォルダの存在確認
  if (!dir.exists(absolute_path)) {
    stop(paste("指定されたフォルダが存在しません:", absolute_path))
  }

  # CSVファイルのみを取得
  files <- list.files(path = absolute_path, pattern = "\\.csv$", full.names = FALSE)
  print(paste("Files Found:", paste(files, collapse = ", ")))

  # プレフィックスに基づいてファイルをフィルタリング
  for (csv_prefix in csv_prefix_list) {
    # プレフィックスが一部一致するファイルを取得
    matched_files <- grep(paste0("...", csv_prefix), files, value = TRUE)

    if (length(matched_files) == 0) {
      warning(paste("プレフィックス「", csv_prefix, "」に一致するファイルが見つかりません。", sep = ""))
      next
    }

    # プレフィックスをキーとしてデータを格納
    if (length(matched_files) == 1) {
      # ファイルが1つの場合
      file_path <- file.path(absolute_path, matched_files)
      data_dict[[csv_prefix]] <- read.csv(file_path, stringsAsFactors = FALSE)
    } else {
      # ファイルが複数ある場合はリストとして格納
      data_list <- lapply(matched_files, function(file) {
        read.csv(file.path(absolute_path, file), stringsAsFactors = FALSE)
      })
      names(data_list) <- matched_files
      data_dict[[csv_prefix]] <- data_list
    }
  }

  return(data_dict)
}


save_png <- function(p, save_path, width = 16, height = 8, create.dir = TRUE) {
  # ディレクトリの確認と作成
  if (create.dir && !dir.exists(dirname(save_path))) {
    dir.create(dirname(save_path), recursive = TRUE)
    print(paste("ディレクトリを作成しました:", dirname(save_path)))
  } else if (!create.dir && !dir.exists(dirname(save_path))) {
    stop(paste("指定されたディレクトリが存在しません:", dirname(save_path)))
  }

  # pngで保存
  ggsave(plot = p,
         filename = save_path,
         width = width,
         height = height
  )
}


# NOTE: htmlに保存する関数
save_plotly_to_html <- function(p, save_path, create.dir = TRUE) {
  # ディレクトリの確認と作成
  if (create.dir && !dir.exists(dirname(save_path))) {
    dir.create(dirname(save_path), recursive = TRUE)
    print(paste("ディレクトリを作成しました:", dirname(save_path)))
  } else if (!create.dir && !dir.exists(dirname(save_path))) {
    stop(paste("指定されたディレクトリが存在しません:", dirname(save_path)))
  }
  # htmlで保存
  # plotly_obj <- ggplotly(p)
  # htmlwidgets::saveWidget(plotly_obj, save_path)
  # ggplotlyの変換を試す
  tryCatch({
    plotly_obj <- ggplotly(p)
    print("ggplotlyの変換に成功しました")
  }, error = function(e) {
    stop(paste("ggplotlyの変換に失敗しました:", e$message))
  })

  # htmlで保存を試す
  tryCatch({
    htmlwidgets::saveWidget(plotly_obj, save_path)
    print(paste("保存に成功しました:", save_path))
  }, error = function(e) {
    stop(paste("保存に失敗しました:", e$message))
  })
}


save_excel <- function(data, file_name, sheet_name, create.dir = TRUE) {
  # ファイルが存在するかチェック
  if (file.exists(file_name)) {
    # 既存のワークブックをロード
    wb <- tryCatch(
      loadWorkbook(file_name),
      error = function(e) {
        stop(paste("既存のExcelファイルをロードできませんでした:", e$message))
      }
    )
    # シート名が既に存在する場合は削除
    if (sheet_name %in% names(wb)) {
      tryCatch(
      {
        removeWorksheet(wb, sheet = sheet_name)
      },
        error = function(e) {
          stop(paste("シート '", sheet_name, "' を削除できませんでした:", e$message))
        }
      )
    }
    addWorksheet(wb, sheet_name)
    writeData(wb, sheet = sheet_name, x = data, rowNames = FALSE)
    saveWorkbook(wb, file = file_name, overwrite = TRUE)
  } else {
    # ディレクトリの確認と作成
    dir_path <- dirname(file_name)  # ファイルパスからディレクトリパスを取得
    if (!dir.exists(dir_path)) {  # ディレクトリが存在しない場合
      dir.create(dir_path, recursive = TRUE)
      print(paste("ディレクトリを作成しました:", dir_path))
    }
    wb <- createWorkbook()
    addWorksheet(wb, sheet_name)
    writeData(wb, sheet = sheet_name, x = data, rowNames = FALSE)
    saveWorkbook(wb, file = file_name, overwrite = TRUE)
  }
}



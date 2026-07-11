import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


DEFAULT_DATA_DIR = "data"
METADATA_FILE = "episode_metadata.json"
TIMESERIES_FILE = "episode_timeseries.json"
BEIJING_TZ = "Asia/Shanghai"
PREVIEW_DURATION_CUTOFF_MS = 10 * 60 * 1000
ENGAGEMENT_FIELDS = ("view", "like", "coin", "favorite", "share", "reply", "dm")
TABLE_COLUMNS = [
    "ep_id",
    "playlist_index",
    "episode_title",
    "long_title",
    "season",
    "duration",
    "created_at",
    "latest_captured_at",
    "view",
    "like",
    "coin",
    "favorite",
    "share",
    "reply",
    "dm",
    "url",
    "cover",
]
ENGAGEMENT_LABELS = {
    "view": "播放",
    "like": "点赞",
    "coin": "投币",
    "favorite": "收藏",
    "share": "分享",
    "reply": "评论",
    "dm": "弹幕",
}
COLUMN_LABELS = {
    "ep_id": "剧集 ID",
    "playlist_index": "序号",
    "episode_title": "集数",
    "long_title": "标题",
    "season": "系列",
    "duration": "时长",
    "created_at": "发布时间",
    "latest_captured_at": "最新采集",
    "view": "播放",
    "like": "点赞",
    "coin": "投币",
    "favorite": "收藏",
    "share": "分享",
    "reply": "评论",
    "dm": "弹幕",
    "url": "链接",
    "cover": "封面",
}
NO_DIVIDER_LABEL = "不除以其他指标"
ALL_TIME_LABEL = "全部时间"
LAST_24_HOURS_LABEL = "最近 24 小时"
LAST_7_DAYS_LABEL = "最近 7 天"
LAST_30_DAYS_LABEL = "最近 30 天"
LAST_90_DAYS_LABEL = "最近 90 天"
CAPTURE_TIME_LABEL = "采集时间"
RELATIVE_TO_CREATION_LABEL = "相对发布时间"
ABSOLUTE_VALUE_LABEL = "累计值"
DELTA_VALUE_LABEL = "较上次采集变化"


def format_metric_label(metric: str, divider_metric: str | None) -> str:
    label = ENGAGEMENT_LABELS.get(metric, metric)
    if divider_metric is None:
        return label
    divider_label = ENGAGEMENT_LABELS.get(divider_metric, divider_metric)
    return f"{label} / {divider_label}"


def apply_metric_divider(df: pd.DataFrame, metrics: list[str], divider_metric: str | None) -> pd.DataFrame:
    if divider_metric is None or divider_metric not in df.columns:
        return df

    ratio_df = df.copy()
    denominator = pd.to_numeric(ratio_df[divider_metric], errors="coerce")
    denominator = denominator.where(denominator != 0)
    for metric in metrics:
        if metric in ratio_df.columns:
            numerator = pd.to_numeric(ratio_df[metric], errors="coerce")
            ratio_df[metric] = numerator / denominator
    return ratio_df


def to_beijing_time(value: Any) -> Any:
    if pd.isna(value):
        return value
    return pd.to_datetime(value, utc=True).tz_convert(BEIJING_TZ)


def format_minute_datetime(value: Any) -> str:
    if pd.isna(value):
        return "None"
    return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")


def derive_season(long_title: Any) -> str:
    if pd.isna(long_title):
        return ""
    season = str(long_title).strip()
    season = re.sub(r"(重制版|重置版)$", "", season)
    season = re.sub(r"\d+$", "", season)
    return season.strip()


def numeric_sort_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def timeseries_field_indexes(timeseries: dict[str, Any]) -> dict[str, int]:
    return {field: index for index, field in enumerate(timeseries["fields"])}


def captured_at_from_row(row: list[Any], field_indexes: dict[str, int]) -> pd.Timestamp:
    return pd.to_datetime(row[field_indexes["captured_at_ms"]], unit="ms", utc=True)


@st.cache_data(ttl=60, show_spinner=False)
def load_latest_rows(data_dir: str) -> pd.DataFrame:
    metadata_path = Path(data_dir) / METADATA_FILE
    timeseries_path = Path(data_dir) / TIMESERIES_FILE
    metadata = read_json_file(metadata_path)
    timeseries = read_json_file(timeseries_path)
    field_indexes = timeseries_field_indexes(timeseries)

    latest_by_ep_id = {}
    for episode_series in timeseries.get("episodes", []):
        ep_id = episode_series.get("ep_id")
        rows = episode_series.get("rows", [])
        if ep_id is None or not rows:
            continue
        latest_by_ep_id[int(ep_id)] = max(rows, key=lambda row: row[field_indexes["captured_at_ms"]])

    rows: list[dict[str, Any]] = []
    for episode in metadata.get("episodes", []):
        ep_id = episode.get("ep_id")
        latest_row = latest_by_ep_id.get(int(ep_id)) if ep_id is not None else None
        rows.append(
            {
                "ep_id": ep_id,
                "playlist_index": episode.get("playlist_index"),
                "episode_title": episode.get("episode_title"),
                "long_title": episode.get("long_title"),
                "season": derive_season(episode.get("long_title")),
                "season_id": episode.get("season_id"),
                "season_title": episode.get("season_title"),
                "bvid": episode.get("bvid"),
                "aid": episode.get("aid"),
                "cid": episode.get("cid"),
                "duration": episode.get("duration"),
                "duration_ms": episode.get("duration_ms"),
                "created_at": to_beijing_time(episode.get("created_at")),
                "pub_time": episode.get("pub_time"),
                "release_date": episode.get("release_date"),
                "latest_captured_at": (
                    captured_at_from_row(latest_row, field_indexes).tz_convert(BEIJING_TZ)
                    if latest_row is not None
                    else None
                ),
                "view": latest_row[field_indexes["view"]] if latest_row is not None else None,
                "like": latest_row[field_indexes["like"]] if latest_row is not None else None,
                "coin": latest_row[field_indexes["coin"]] if latest_row is not None else None,
                "favorite": latest_row[field_indexes["favorite"]] if latest_row is not None else None,
                "share": latest_row[field_indexes["share"]] if latest_row is not None else None,
                "reply": latest_row[field_indexes["reply"]] if latest_row is not None else None,
                "dm": latest_row[field_indexes["dm"]] if latest_row is not None else None,
                "url": episode.get("url"),
                "cover": episode.get("cover"),
                "first_seen_at": episode.get("first_seen_at"),
                "updated_at": episode.get("updated_at"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_playlist_index_sort"] = numeric_sort_series(df["playlist_index"])
    df["_episode_title_sort"] = numeric_sort_series(df["episode_title"])
    return (
        df.sort_values(
            ["season_id", "_playlist_index_sort", "_episode_title_sort", "ep_id"],
            na_position="last",
        )
        .drop(columns=["_playlist_index_sort", "_episode_title_sort"])
        .reset_index(drop=True)
    )


@st.cache_data(ttl=60, show_spinner=False)
def load_time_series(data_dir: str, ep_ids: list[int], since: datetime | None) -> pd.DataFrame:
    if not ep_ids:
        return pd.DataFrame()

    timeseries_path = Path(data_dir) / TIMESERIES_FILE
    timeseries = read_json_file(timeseries_path)
    field_indexes = timeseries_field_indexes(timeseries)
    selected_ep_ids = {int(ep_id) for ep_id in ep_ids}
    rows = []
    for episode_series in timeseries.get("episodes", []):
        ep_id = episode_series.get("ep_id")
        if ep_id is None or int(ep_id) not in selected_ep_ids:
            continue
        for timeseries_row in episode_series.get("rows", []):
            captured_at = captured_at_from_row(timeseries_row, field_indexes)
            if since is not None and captured_at.to_pydatetime() < since:
                continue
            row = {
                "captured_at": captured_at,
                "ep_id": int(ep_id),
            }
            for field in ENGAGEMENT_FIELDS:
                row[field] = timeseries_row[field_indexes[field]]
            rows.append(row)
    return pd.DataFrame(rows)


def format_episode_label(row: pd.Series) -> str:
    title = row.get("long_title") or row.get("episode_title") or ""
    playlist_index = row.get("playlist_index")
    ep_id = row.get("ep_id", row.name)
    if pd.notna(playlist_index):
        return f"{int(playlist_index):03d} | ep{int(ep_id)} | {title}"
    return f"ep{int(ep_id)} | {title}"


def format_time_series_episode_label(row: pd.Series) -> str:
    episode_title = row.get("episode_title")
    long_title = row.get("long_title")
    if pd.notna(episode_title) and pd.notna(long_title):
        return f"{episode_title} | {long_title}"
    if pd.notna(long_title):
        return str(long_title)
    if pd.notna(episode_title):
        return str(episode_title)
    ep_id = row.get("ep_id", row.name)
    return f"ep{int(ep_id)}"


def build_chart_df(
    series_df: pd.DataFrame,
    latest_df: pd.DataFrame,
    metrics: list[str],
    time_axis_mode: str,
    value_mode: str,
    divider_metric: str | None,
) -> pd.DataFrame:
    if series_df.empty:
        return series_df

    labels = latest_df.set_index("ep_id").apply(format_time_series_episode_label, axis=1).to_dict()
    created_at_by_ep_id = latest_df.set_index("ep_id")["created_at"].to_dict()
    value_df = apply_metric_divider(series_df, metrics, divider_metric)
    chart_df = value_df.melt(
        id_vars=["captured_at", "ep_id"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    chart_df["episode"] = chart_df["ep_id"].map(labels).fillna(chart_df["ep_id"].astype(str))
    chart_df["metric"] = chart_df["metric"].apply(lambda metric: format_metric_label(metric, divider_metric))
    chart_df["captured_at"] = pd.to_datetime(chart_df["captured_at"], utc=True).dt.tz_convert(BEIJING_TZ)
    if value_mode == DELTA_VALUE_LABEL:
        chart_df = chart_df.sort_values(["ep_id", "metric", "captured_at"])
        chart_df["value"] = chart_df.groupby(["ep_id", "metric"])["value"].diff()
        chart_df = chart_df.dropna(subset=["value"])
    if time_axis_mode == RELATIVE_TO_CREATION_LABEL:
        chart_df["created_at"] = chart_df["ep_id"].map(created_at_by_ep_id)
        chart_df["created_at"] = pd.to_datetime(chart_df["created_at"], utc=True)
        chart_df["days_since_created"] = (
            chart_df["captured_at"] - chart_df["created_at"]
        ).dt.total_seconds() / 86400
        chart_df = chart_df.dropna(subset=["days_since_created"])
    return chart_df.dropna(subset=["value"])


def episode_title_sort_value(value: Any) -> tuple[int, Any]:
    if pd.isna(value):
        return (1, "")
    text = str(value).strip()
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def sorted_episode_title_order(df: pd.DataFrame) -> list[str]:
    order_df = df[["episode_title_display", "episode_sort", "playlist_index_sort", "ep_id"]].drop_duplicates()
    order_df = order_df.sort_values(["episode_sort", "playlist_index_sort", "ep_id"], na_position="last")
    return order_df["episode_title_display"].drop_duplicates().tolist()


def build_latest_engagement_df(
    latest_df: pd.DataFrame,
    metrics: list[str],
    divider_metric: str | None,
) -> pd.DataFrame:
    if latest_df.empty or not metrics:
        return pd.DataFrame()

    overview_df = latest_df.copy()
    overview_df["episode_label"] = overview_df.apply(format_episode_label, axis=1)
    overview_df["episode_sort"] = overview_df["episode_title"].apply(episode_title_sort_value)
    overview_df["playlist_index_sort"] = numeric_sort_series(overview_df["playlist_index"])
    overview_df["episode_title_display"] = overview_df["episode_title"].astype(str)
    episode_order = sorted_episode_title_order(overview_df)
    value_df = apply_metric_divider(overview_df, metrics, divider_metric)

    chart_df = value_df.melt(
        id_vars=["episode_label", "episode_title_display", "long_title", "season", "playlist_index", "ep_id"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    chart_df["metric"] = chart_df["metric"].apply(lambda metric: format_metric_label(metric, divider_metric))
    chart_df["episode_title_display"] = pd.Categorical(
        chart_df["episode_title_display"],
        categories=episode_order,
        ordered=True,
    )
    return chart_df.dropna(subset=["value"])


def filter_episode_rows(latest_df: pd.DataFrame, hide_previews: bool) -> pd.DataFrame:
    if not hide_previews or "duration_ms" not in latest_df.columns:
        return latest_df
    return latest_df[latest_df["duration_ms"].fillna(0) >= PREVIEW_DURATION_CUTOFF_MS].copy()


def normalize_display_sort_columns(display_df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = display_df.copy()
    for column in ("playlist_index", "episode_title"):
        if column not in normalized_df.columns:
            continue
        numeric_values = pd.to_numeric(normalized_df[column], errors="coerce")
        has_value = normalized_df[column].notna()
        if numeric_values[has_value].notna().all():
            normalized_df[column] = numeric_values.astype("Int64")
    return normalized_df


def render_dashboard() -> None:
    st.set_page_config(page_title="B 站剧集数据", layout="wide")
    st.title("B 站剧集数据")
    data_dir = DEFAULT_DATA_DIR

    with st.sidebar:
        st.header("筛选")
        hide_previews = st.checkbox("隐藏短于 10 分钟的预告片", value=True)
        st.header("图表")
        metric_labels = [ENGAGEMENT_LABELS[field] for field in ENGAGEMENT_FIELDS]
        selected_metric_labels = st.multiselect("互动指标", metric_labels, default=["播放"])
        selected_metrics = [
            field for field, label in ENGAGEMENT_LABELS.items() if label in selected_metric_labels
        ]
        divider_options = [NO_DIVIDER_LABEL] + metric_labels
        selected_divider_label = st.selectbox("指标除数", divider_options)
        divider_metric = None
        if selected_divider_label != NO_DIVIDER_LABEL:
            divider_metric = next(
                field for field, label in ENGAGEMENT_LABELS.items() if label == selected_divider_label
            )
        time_window = st.selectbox(
            "时间范围",
            (ALL_TIME_LABEL, LAST_24_HOURS_LABEL, LAST_7_DAYS_LABEL, LAST_30_DAYS_LABEL, LAST_90_DAYS_LABEL),
        )
        time_axis_mode = st.radio(
            "时间序列横轴",
            (CAPTURE_TIME_LABEL, RELATIVE_TO_CREATION_LABEL),
        )
        value_mode = st.radio(
            "时间序列数值",
            (ABSOLUTE_VALUE_LABEL, DELTA_VALUE_LABEL),
        )
        if st.button("刷新数据", use_container_width=True):
            st.cache_data.clear()

    try:
        latest_df = load_latest_rows(data_dir)
    except FileNotFoundError as exc:
        st.error(f"缺少本地数据文件：{exc.filename}")
        return
    except Exception as exc:
        st.error(f"无法加载本地 JSON 数据：{exc}")
        return

    if latest_df.empty:
        st.warning("没有找到剧集数据。请先运行 dump_mongo_to_json.py 填充 data 文件夹。")
        return

    filtered_df = filter_episode_rows(latest_df, hide_previews)
    total_episodes = len(filtered_df)
    hidden_episode_count = len(latest_df) - len(filtered_df)
    latest_capture = latest_df["latest_captured_at"].max()
    season_title = latest_df["season_title"].dropna().iloc[0] if latest_df["season_title"].notna().any() else ""
    summary_cols = st.columns(3)
    summary_cols[0].metric("剧集数", f"{total_episodes:,}")
    summary_cols[1].metric("最新采集", format_minute_datetime(latest_capture))
    summary_cols[2].metric("系列", season_title or "未知")
    if hide_previews and hidden_episode_count:
        st.caption(f"已隐藏 {hidden_episode_count:,} 条短于 10 分钟的预告片记录。")

    st.subheader("剧集")
    display_df = filtered_df[[column for column in TABLE_COLUMNS if column in filtered_df.columns]].copy()
    display_df = normalize_display_sort_columns(display_df)
    display_df.insert(0, "选择", False)
    episode_title_column = (
        st.column_config.NumberColumn(COLUMN_LABELS["episode_title"])
        if "episode_title" in display_df.columns and pd.api.types.is_numeric_dtype(display_df["episode_title"])
        else st.column_config.TextColumn(COLUMN_LABELS["episode_title"])
    )
    edited_df = st.data_editor(
        display_df,
        hide_index=True,
        use_container_width=True,
        height=430,
        disabled=[column for column in display_df.columns if column != "选择"],
        column_config={
            "选择": st.column_config.CheckboxColumn("选择"),
            "ep_id": st.column_config.NumberColumn(COLUMN_LABELS["ep_id"]),
            "playlist_index": st.column_config.NumberColumn(COLUMN_LABELS["playlist_index"]),
            "episode_title": episode_title_column,
            "long_title": st.column_config.TextColumn(COLUMN_LABELS["long_title"]),
            "season": st.column_config.TextColumn(COLUMN_LABELS["season"]),
            "duration": st.column_config.TextColumn(COLUMN_LABELS["duration"]),
            "created_at": st.column_config.DatetimeColumn(COLUMN_LABELS["created_at"]),
            "latest_captured_at": st.column_config.DatetimeColumn(COLUMN_LABELS["latest_captured_at"]),
            "view": st.column_config.NumberColumn(COLUMN_LABELS["view"]),
            "like": st.column_config.NumberColumn(COLUMN_LABELS["like"]),
            "coin": st.column_config.NumberColumn(COLUMN_LABELS["coin"]),
            "favorite": st.column_config.NumberColumn(COLUMN_LABELS["favorite"]),
            "share": st.column_config.NumberColumn(COLUMN_LABELS["share"]),
            "reply": st.column_config.NumberColumn(COLUMN_LABELS["reply"]),
            "dm": st.column_config.NumberColumn(COLUMN_LABELS["dm"]),
            "url": st.column_config.LinkColumn(COLUMN_LABELS["url"]),
            "cover": st.column_config.LinkColumn(COLUMN_LABELS["cover"]),
        },
        key="episodes_grid",
    )
    selected_ep_ids = edited_df.loc[edited_df["选择"], "ep_id"].dropna().astype(int).tolist()

    st.subheader("各剧集最新互动数据")
    overview_chart_df = build_latest_engagement_df(filtered_df, selected_metrics, divider_metric)
    if overview_chart_df.empty:
        st.info("请在侧边栏至少选择一个互动指标。")
    else:
        overview_fig = px.line(
            overview_chart_df,
            x="episode_title_display",
            y="value",
            color="season",
            facet_row="metric" if len(selected_metrics) > 1 else None,
            line_group="season",
            markers=True,
            hover_data=["long_title", "ep_id"],
            labels={
                "episode_title_display": "剧集",
                "value": "最新比值" if divider_metric else "最新数值",
                "season": "系列",
                "long_title": "标题",
                "ep_id": "剧集 ID",
            },
            category_orders={"episode_title_display": overview_chart_df["episode_title_display"].cat.categories.tolist()},
            height=max(420, 230 * len(selected_metrics)),
        )
        overview_fig.update_layout(legend_title_text="系列", hovermode="x unified")
        overview_fig.update_xaxes(type="category", categoryorder="array", categoryarray=overview_chart_df["episode_title_display"].cat.categories.tolist())
        overview_fig.update_yaxes(matches=None)
        st.plotly_chart(overview_fig, use_container_width=True)

    st.subheader("互动数据时间序列")
    if not selected_ep_ids:
        st.info("请在表格中选择一个或多个剧集来绘制时间序列。")
        return
    if not selected_metrics:
        st.info("请在侧边栏至少选择一个互动指标。")
        return

    now = datetime.now(timezone.utc)
    since_by_window = {
        ALL_TIME_LABEL: None,
        LAST_24_HOURS_LABEL: now - timedelta(days=1),
        LAST_7_DAYS_LABEL: now - timedelta(days=7),
        LAST_30_DAYS_LABEL: now - timedelta(days=30),
        LAST_90_DAYS_LABEL: now - timedelta(days=90),
    }
    series_df = load_time_series(data_dir, selected_ep_ids, since_by_window[time_window])
    chart_df = build_chart_df(series_df, filtered_df, selected_metrics, time_axis_mode, value_mode, divider_metric)
    if chart_df.empty:
        st.warning("在所选剧集和时间范围内没有找到采集记录。")
        return

    x_axis = "captured_at"
    x_axis_label = "采集时间"
    y_axis_label = "比值" if divider_metric else "数值"
    if time_axis_mode == RELATIVE_TO_CREATION_LABEL:
        x_axis = "days_since_created"
        x_axis_label = "发布后天数"
    if value_mode == DELTA_VALUE_LABEL:
        y_axis_label = "较上次采集的比值变化" if divider_metric else "较上次采集变化"

    fig = px.line(
        chart_df,
        x=x_axis,
        y="value",
        color="episode",
        facet_row="metric" if len(selected_metrics) > 1 else None,
        markers=True,
        labels={x_axis: x_axis_label, "value": y_axis_label, "episode": "剧集"},
        height=max(420, 250 * len(selected_metrics)),
    )
    fig.update_layout(legend_title_text="剧集", hovermode="x unified")
    fig.update_yaxes(matches=None)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    render_dashboard()

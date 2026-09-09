let chatId = crypto.randomUUID();
let effort = "medium";

const $ = (id) =>
  document.getElementById(id);

const chat = $("chat");
const input = $("input");
const send = $("send");
const train = $("train");
const stop = $("stop");

function addMessage(
  role,
  text,
) {
  const element =
    document.createElement("div");

  element.className =
    `msg ${role}`;

  const avatar =
    role === "user"
      ? "Y"
      : "V";

  element.innerHTML = `
    <div class="avatar">
      ${avatar}
    </div>
    <div class="bubble"></div>
  `;

  element
    .querySelector(".bubble")
    .textContent = text;

  chat.appendChild(
    element
  );

  chat.scrollTop =
    chat.scrollHeight;
}


async function refreshStatus() {

  try {

    const response =
      await fetch(
        "/api/status",
        {
          cache: "no-store",
        }
      );

    if (!response.ok) {
      throw new Error(
        "Status request failed"
      );
    }

    const data =
      await response.json();

    $("steps").textContent =
      Number(
        data.model_step || 0
      ).toLocaleString();

    $("params").textContent =
      Number(
        data.parameters || 0
      ).toLocaleString();

    const status =
      data.training || {};

    if (status.running) {

      train.disabled = true;

      const parts = [
        status.phase || "training",
      ];

      if (
        Number.isFinite(
          status.elapsed
        )
      ) {
        parts.push(
          `${Math.floor(
            status.elapsed
          )}s`
        );
      }

      if (
        Number.isFinite(
          status.steps
        )
      ) {
        parts.push(
          `${status.steps.toLocaleString()} steps`
        );
      }

      if (
        Number.isFinite(
          status.train_loss
        )
      ) {
        parts.push(
          `loss ${status.train_loss.toFixed(3)}`
        );
      }

      if (
        Number.isFinite(
          status.val_loss
        )
      ) {
        parts.push(
          `val ${status.val_loss.toFixed(3)}`
        );
      }

      $("status").textContent =
        parts.join(" • ");

    } else {

      train.disabled = false;

      if (
        status.phase === "complete"
      ) {

        const mastery =
          Number.isFinite(
            status.mastery
          )
            ? ` • evaluation ${(status.mastery * 100).toFixed(1)}%`
            : "";

        $("status").textContent =
          `Complete • ${status.steps || 0} steps${mastery}`;

      } else {

        $("status").textContent =
          status.message ||
          "Idle";
      }
    }

    $("conn").textContent =
      "● connected";

    $("conn").style.color =
      "#7bd992";

  } catch (error) {

    $("conn").textContent =
      "● offline";

    $("conn").style.color =
      "#f28b82";

  }
}


async function sendMessage() {

  const message =
    input.value.trim();

  if (!message) {
    return;
  }

  input.value = "";

  addMessage(
    "user",
    message,
  );

  send.disabled = true;

  try {

    const response =
      await fetch(
        "/api/chat",
        {
          method: "POST",
          headers: {
            "content-type":
              "application/json",
          },
          body: JSON.stringify({
            message,
            chat_id: chatId,
            effort,
          }),
        }
      );

    const data =
      await response.json();

    if (!response.ok) {
      throw new Error(
        data.detail ||
        "Chat request failed"
      );
    }

    addMessage(
      "assistant",
      data.response || "",
    );

  } catch (error) {

    addMessage(
      "assistant",
      `Backend error: ${error.message}`,
    );

  } finally {

    send.disabled = false;
    input.focus();

  }
}


send.onclick =
  sendMessage;


input.addEventListener(
  "keydown",
  (event) => {

    if (
      event.key === "Enter"
      && !event.shiftKey
    ) {

      event.preventDefault();

      sendMessage();
    }
  }
);


document
  .querySelectorAll(
    ".effort button"
  )
  .forEach(
    (button) => {

      button.onclick = () => {

        document
          .querySelectorAll(
            ".effort button"
          )
          .forEach(
            (item) =>
              item.classList.remove(
                "active"
              )
          );

        button.classList.add(
          "active"
        );

        effort =
          button.dataset.e;
      };
    }
  );


$("new").onclick =
  () => {

    chatId =
      crypto.randomUUID();

    chat.innerHTML = "";

    addMessage(
      "assistant",
      "New chat ready.",
    );
  };


train.onclick =
  async () => {

    const goal =
      $("goal").value.trim();

    const minutes =
      Number(
        $("mins").value
      ) || 20;

    if (!goal) {

      $("status").textContent =
        "Enter a training goal.";

      return;
    }

    train.disabled = true;

    try {

      const response =
        await fetch(
          "/api/train/start",
          {
            method: "POST",
            headers: {
              "content-type":
                "application/json",
            },
            body:
              JSON.stringify({
                goal,
                minutes,
              }),
          }
        );

      const data =
        await response.json();

      if (!response.ok) {

        throw new Error(
          data.detail ||
          "Training failed to start"
        );
      }

      $("status").textContent =
        `Training started • ${goal}`;

    } catch (error) {

      $("status").textContent =
        error.message;

      train.disabled = false;
    }
  };


stop.onclick =
  async () => {

    try {

      await fetch(
        "/api/train/stop",
        {
          method: "POST",
        }
      );

    } catch (error) {

      $("status").textContent =
        error.message;
    }
  };


addMessage(
  "assistant",
  "Hello. I am Verdant-1.0.",
);


setInterval(
  refreshStatus,
  1000,
);


refreshStatus();

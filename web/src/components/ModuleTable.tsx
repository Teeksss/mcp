// ...
import { useAuth } from "../context/AuthContext";
// ...
export default function ModuleTable() {
  // ...
  const { role } = useAuth();
  // ...
  return (
    <div>
      {role === "admin" && <ModuleForm onSuccess={() => setRefresh(!refresh)} />}
      {/* ... */}
      <Table
        // ...
        columns={[
          // ...
          role === "admin"
            ? {
                title: "İşlem",
                render: (_, rec) => (
                  <>
                    <Button size="small" onClick={() => setEditing(rec)}>
                      Düzenle
                    </Button>
                    <Popconfirm title="Silinsin mi?" onConfirm={() => handleDelete(rec.id)}>
                      <Button size="small" danger>
                        SİL
                      </Button>
                    </Popconfirm>
                  </>
                ),
              }
            : {},
        ]}
      />
      {/* ... */}
    </div>
  );
}